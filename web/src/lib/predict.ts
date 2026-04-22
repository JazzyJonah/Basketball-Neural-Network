import * as ort from 'onnxruntime-web';
import { fetchJson, assetUrl } from './fetchJson';

type SnapshotRow = {
  date: string;
  teamId: number;
  features: number[];
};

type SnapshotSeasonFile = {
  season: number;
  featureNames: string[];
  rows: SnapshotRow[];
};

type RawModelMeta = {
  team_id_to_index?: Record<string, number>;
  teamIdToIndex?: Record<string, number>;
  means?: number[];
  stds?: number[];
  featureCols?: string[];
  feature_cols?: string[];
  inputNames?: string[];
  input_names?: string[];
  outputNames?: string[];
  output_names?: string[];
};

export type ModelMeta = {
  teamIdToIndex: Record<string, number>;
  means: number[];
  stds: number[];
  featureCols: string[];
  inputNames: string[];
  outputNames: string[];
};

export type PredictInputs = {
  season: number;
  gameDate: string;
  team1Id: number;
  team2Id: number;
  homeIndicator: number;
};

export type PredictResult = {
  team1Score: number;
  team2Score: number;
};

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let modelMetaPromise: Promise<ModelMeta> | null = null;
const snapshotCache = new Map<number, Promise<SnapshotSeasonFile>>();

async function getSession(): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(assetUrl('model/model.onnx'), {
      executionProviders: ['wasm'],
    });
  }
  return sessionPromise;
}

function normalizeModelMeta(raw: RawModelMeta): ModelMeta {
  const teamIdToIndex = raw.team_id_to_index ?? raw.teamIdToIndex;
  const means = raw.means ?? [];
  const stds = raw.stds ?? [];
  const featureCols = raw.featureCols ?? raw.feature_cols ?? [];
  const inputNames = raw.inputNames ?? raw.input_names ?? ['x_cont', 'team1_id', 'team2_id'];
  const outputNames = raw.outputNames ?? raw.output_names ?? ['pred_scores'];

  if (!teamIdToIndex) {
    throw new Error('model_meta.json is missing team_id_to_index/teamIdToIndex');
  }
  if (!means.length || !stds.length) {
    throw new Error('model_meta.json is missing means/stds');
  }

  return {
    teamIdToIndex,
    means,
    stds,
    featureCols,
    inputNames,
    outputNames,
  };
}

async function getModelMeta(): Promise<ModelMeta> {
  if (!modelMetaPromise) {
    modelMetaPromise = fetchJson<RawModelMeta>('model/model_meta.json').then(normalizeModelMeta);
  }
  return modelMetaPromise;
}

async function getSeasonSnapshot(season: number): Promise<SnapshotSeasonFile> {
  if (!snapshotCache.has(season)) {
    snapshotCache.set(season, fetchJson<SnapshotSeasonFile>(`data/snapshots/${season}.json`));
  }
  return snapshotCache.get(season)!;
}

function findTeamSnapshot(
  snapshotSeason: SnapshotSeasonFile,
  gameDate: string,
  teamId: number
): SnapshotRow | null {
  for (const row of snapshotSeason.rows) {
    if (row.date === gameDate && row.teamId === teamId) {
      return row;
    }
  }
  return null;
}

function standardizeVector(raw: Float32Array, means: number[], stds: number[]): Float32Array {
  if (raw.length !== means.length || raw.length !== stds.length) {
    throw new Error(
      `Standardization length mismatch: raw=${raw.length}, means=${means.length}, stds=${stds.length}`
    );
  }

  const out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    const std = stds[i] === 0 ? 1 : stds[i];
    out[i] = (raw[i] - means[i]) / std;
  }
  return out;
}

function hasBadValues(arr: Float32Array): boolean {
  for (const v of arr) {
    if (!Number.isFinite(v)) return true;
    if (Math.abs(v) > 1e6) return true;
  }
  return false;
}

function resolveInputNames(session: ort.InferenceSession, meta: ModelMeta) {
  const inputNames = session.inputNames;

  const contName =
    inputNames.find((name) => meta.inputNames.includes(name)) ??
    inputNames.find((name) => ['x_cont', 'x', 'features', 'continuous_x'].includes(name)) ??
    inputNames.find((name) => name.toLowerCase().includes('cont')) ??
    inputNames[0];

  const team1Name =
    inputNames.find((name) => ['team1_id', 'team1id', 'team1'].includes(name)) ??
    inputNames.find((name) => name.toLowerCase().includes('team1')) ??
    inputNames[1];

  const team2Name =
    inputNames.find((name) => ['team2_id', 'team2id', 'team2'].includes(name)) ??
    inputNames.find((name) => name.toLowerCase().includes('team2')) ??
    inputNames[2];

  if (!contName || !team1Name || !team2Name) {
    throw new Error(`Could not resolve ONNX input names. Found: ${inputNames.join(', ')}`);
  }

  return { contName, team1Name, team2Name };
}

export async function predictHistoricalGame(inputs: PredictInputs): Promise<PredictResult> {
  const [session, meta, snapshotSeason] = await Promise.all([
    getSession(),
    getModelMeta(),
    getSeasonSnapshot(inputs.season),
  ]);

  const team1 = findTeamSnapshot(snapshotSeason, inputs.gameDate, inputs.team1Id);
  const team2 = findTeamSnapshot(snapshotSeason, inputs.gameDate, inputs.team2Id);

  if (!team1) {
    throw new Error(
      `No snapshot found for team ${inputs.team1Id} on ${inputs.gameDate} in season ${inputs.season}`
    );
  }

  if (!team2) {
    throw new Error(
      `No snapshot found for team ${inputs.team2Id} on ${inputs.gameDate} in season ${inputs.season}`
    );
  }

  if (team1.features.length !== team2.features.length) {
    throw new Error('Team feature lengths do not match');
  }

  const diff = team1.features.map((v, i) => v - team2.features[i]);
  const sum = team1.features.map((v, i) => v + team2.features[i]);

  const xContRaw = new Float32Array([
    ...team1.features,
    ...team2.features,
    ...diff,
    ...sum,
    inputs.homeIndicator,
  ]);

  if (hasBadValues(xContRaw)) {
    throw new Error('Raw input feature vector contains invalid or extremely large values');
  }

  const xCont = standardizeVector(xContRaw, meta.means, meta.stds);

  if (hasBadValues(xCont)) {
    throw new Error('Standardized input feature vector contains invalid or extremely large values');
  }

  const team1Index = meta.teamIdToIndex[String(inputs.team1Id)];
  const team2Index = meta.teamIdToIndex[String(inputs.team2Id)];

  if (team1Index === undefined) {
    throw new Error(`Missing embedding index for team ${inputs.team1Id}`);
  }
  if (team2Index === undefined) {
    throw new Error(`Missing embedding index for team ${inputs.team2Id}`);
  }

  const { contName, team1Name, team2Name } = resolveInputNames(session, meta);

  console.log('team1 snapshot:', team1);
  console.log('team2 snapshot:', team2);
  console.log('raw feature lengths:', {
    team1: team1.features.length,
    team2: team2.features.length,
  });
  console.log('raw xCont preview:', Array.from(xContRaw.slice(0, 20)));
  console.log('standardized xCont preview:', Array.from(xCont.slice(0, 20)));
  console.log('homeIndicator:', inputs.homeIndicator);
  console.log('team embedding indices:', {
    team1Id: inputs.team1Id,
    team1Index,
    team2Id: inputs.team2Id,
    team2Index,
  });
  console.log('onnx input names:', session.inputNames);
  console.log('onnx output names:', session.outputNames);

  const feeds: Record<string, ort.Tensor> = {
    [contName]: new ort.Tensor('float32', xCont, [1, xCont.length]),
    [team1Name]: new ort.Tensor('int64', BigInt64Array.from([BigInt(team1Index)]), [1]),
    [team2Name]: new ort.Tensor('int64', BigInt64Array.from([BigInt(team2Index)]), [1]),
  };

  const results = await session.run(feeds);
  const outputName = Object.keys(results)[0];
  const output = results[outputName].data;

  console.log('raw model output:', output);

  return {
    team1Score: Number(output[0]),
    team2Score: Number(output[1]),
  };
}