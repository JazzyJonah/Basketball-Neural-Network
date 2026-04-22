import { FrontendModelMetadata, SeasonGames, SeasonSnapshots, TeamMap } from '../types'

export async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(path)
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path}: ${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export function availableSeasonsFromGamesFiles(): number[] {
  return Array.from({ length: 40 }, (_, i) => 2001 + i)
}

export async function loadTeams(): Promise<TeamMap> {
  return fetchJson<TeamMap>('./data/teams.json')
}

export async function loadSeasonGames(season: number): Promise<SeasonGames> {
  return fetchJson<SeasonGames>(`./data/games/${season}.json`)
}

export async function loadSeasonSnapshots(season: number): Promise<SeasonSnapshots> {
  return fetchJson<SeasonSnapshots>(`./data/snapshots/${season}.json`)
}

export async function loadModelMetadata(): Promise<FrontendModelMetadata> {
  return fetchJson<FrontendModelMetadata>('./model/model_meta.json')
}
