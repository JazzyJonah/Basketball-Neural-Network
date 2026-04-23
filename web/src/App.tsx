import { useEffect, useMemo, useState } from 'react';
import './app.css';
import { fetchJson } from './lib/fetchJson';
import { predictHistoricalGame } from './lib/predict';

type TeamMap = Record<string, { id: number; displayName: string; shortName?: string; abbreviation?: string }>;

type GameRecord = {
  id: string;
  season: number;
  date: string;
  team1Id: number;
  team1Name: string;
  team2Id: number;
  team2Name: string;
  team1Home: boolean;
  team2Home: boolean;
  team1Score: number;
  team2Score: number;
  team1WinsBefore?: number;
  team1LossesBefore?: number;
  team2WinsBefore?: number;
  team2LossesBefore?: number;
};

type PredictionResult = {
  team1Score: number;
  team2Score: number;
};

type LogLevel = 'info' | 'warn' | 'error';

type LogEvent = {
  level: LogLevel;
  message: string;
  timestamp: string;
};

function formatMatchup(game: GameRecord) {
  const neutral = !game.team1Home && !game.team2Home;
  if (neutral) return `${game.team1Name} vs ${game.team2Name}`;
  if (game.team1Home) return `${game.team2Name} @ ${game.team1Name}`;
  return `${game.team1Name} @ ${game.team2Name}`;
}

function formatScoreLine(game: GameRecord) {
  const neutral = !game.team1Home && !game.team2Home;
  if (neutral || game.team1Home) {
    return `${game.team1Name} ${game.team1Score} - ${game.team2Name} ${game.team2Score}`;
  }
  return `${game.team2Name} ${game.team2Score} - ${game.team1Name} ${game.team1Score}`;
}

function formatPredictedScoreLine(game: GameRecord, prediction: PredictionResult | null) {
  if (!prediction) return '—';
  const neutral = !game.team1Home && !game.team2Home;
  if (neutral || game.team1Home) {
    return `${game.team1Name} ${prediction.team1Score.toFixed(1)} - ${game.team2Name} ${prediction.team2Score.toFixed(1)}`;
  }
  return `${game.team2Name} ${prediction.team2Score.toFixed(1)} - ${game.team1Name} ${prediction.team1Score.toFixed(1)}`;
}

function predictedWinner(game: GameRecord, prediction: PredictionResult | null) {
  if (!prediction) return '';
  return prediction.team1Score >= prediction.team2Score ? game.team1Name : game.team2Name;
}

function actualWinner(game: GameRecord) {
  return game.team1Score >= game.team2Score ? game.team1Name : game.team2Name;
}

function winnerCorrect(game: GameRecord, prediction: PredictionResult | null) {
  if (!prediction) return '';
  return predictedWinner(game, prediction) === actualWinner(game) ? 'Yes' : 'No';
}

function rmseScore(game: GameRecord, prediction: PredictionResult | null) {
  if (!prediction) return '';
  const predSpread = prediction.team1Score - prediction.team2Score;
  const actualSpread = game.team1Score - game.team2Score;
  const mse = (
    Math.pow(prediction.team1Score - game.team1Score, 2) +
    Math.pow(prediction.team2Score - game.team2Score, 2) +
    Math.pow(predSpread - actualSpread, 2)
  ) / 3;
  return Math.sqrt(mse).toFixed(2);
}

function formatRecord(wins?: number, losses?: number) {
  if (wins == null || losses == null) return '—';
  return `(${wins}-${losses})`;
}

export default function App() {
  const [teams, setTeams] = useState<TeamMap>({});
  const [games, setGames] = useState<GameRecord[]>([]);
  const [selectedSeason, setSelectedSeason] = useState<string>('');
  const [selectedTeamId, setSelectedTeamId] = useState<string>('');
  const [selectedGameId, setSelectedGameId] = useState<string>('');
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [loadingData, setLoadingData] = useState(true);
  const [error, setError] = useState<string>('');
  const [logs, setLogs] = useState<LogEvent[]>([]);

  const addLog = (level: LogLevel, message: string) => {
    setLogs((prev) => [{ level, message, timestamp: new Date().toLocaleTimeString() }, ...prev].slice(0, 25));
    if (level === 'error') console.error(message);
    else if (level === 'warn') console.warn(message);
    else console.log(message);
  };

  useEffect(() => {
    (async () => {
      setLoadingData(true);
      try {
        addLog('info', 'Loading teams and historical games');

        const [teamData, gameData] = await Promise.all([
          fetchJson<TeamMap>('data/teams.json'),
          fetchJson<GameRecord[]>('data/games/all_games.json'),
        ]);

        setTeams(teamData);
        setGames(gameData);

        const seasons = [...new Set(gameData.map((g) => g.season))].sort((a, b) => b - a);
        if (seasons.length > 0) {
          setSelectedSeason(String(seasons[0]));
        }

        addLog('info', `Loaded ${Object.keys(teamData).length} teams`);
        addLog('info', `Loaded ${gameData.length} historical games`);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown load error';
        setError(message);
        addLog('error', message);
      } finally {
        setLoadingData(false);
      }
    })();
  }, []);

  const seasons = useMemo(
    () => [...new Set(games.map((g) => g.season))].sort((a, b) => b - a),
    [games]
  );

  const seasonGames = useMemo(
    () => games.filter((g) => String(g.season) === selectedSeason),
    [games, selectedSeason]
  );

  const teamOptions = useMemo(() => {
    const map = new Map<number, string>();
    for (const game of seasonGames) {
      map.set(game.team1Id, game.team1Name);
      map.set(game.team2Id, game.team2Name);
    }
    return [...map.entries()]
      .map(([id, name]) => ({ id, name }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [seasonGames]);

  const filteredGames = useMemo(() => {
    if (!selectedTeamId) return seasonGames;
    const tid = Number(selectedTeamId);
    return seasonGames.filter((g) => g.team1Id === tid || g.team2Id === tid);
  }, [seasonGames, selectedTeamId]);

  const selectedGame = useMemo(
    () => filteredGames.find((game) => game.id === selectedGameId) ?? null,
    [filteredGames, selectedGameId]
  );

  useEffect(() => {
    if (teamOptions.length > 0 && selectedTeamId) {
      const exists = teamOptions.some((t) => String(t.id) === selectedTeamId);
      if (!exists) {
        setSelectedTeamId('');
      }
    }
  }, [teamOptions, selectedTeamId]);

  useEffect(() => {
    if (filteredGames.length > 0) {
      setSelectedGameId((prev) => {
        const exists = filteredGames.some((g) => g.id === prev);
        return exists ? prev : filteredGames[0].id;
      });
    } else {
      setSelectedGameId('');
    }
  }, [filteredGames]);

  useEffect(() => {
    setPrediction(null);
    setError('');
  }, [selectedGameId]);

  const onPredict = async () => {
    if (!selectedGame) return;

    setLoadingPrediction(true);
    setError('');
    addLog('info', `Running historical prediction for game ${selectedGame.id}`);

    try {
      const homeIndicator =
        !selectedGame.team1Home && !selectedGame.team2Home
          ? 0
          : selectedGame.team1Home
            ? 1
            : -1;

      const result = await predictHistoricalGame({
        season: selectedGame.season,
        gameDate: selectedGame.date,
        team1Id: selectedGame.team1Id,
        team2Id: selectedGame.team2Id,
        homeIndicator,
      });

      setPrediction(result);
      addLog('info', `Prediction completed for ${formatMatchup(selectedGame)}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Prediction failed';
      setError(message);
      setPrediction(null);
      addLog('error', message);
    } finally {
      setLoadingPrediction(false);
    }
  };

  return (
    <div className="page-shell">
      <header className="hero-card">
        <div>
          <p className="eyebrow">Historical Game Predictor</p>
          <h1>NCAA men&apos;s basketball backtest viewer</h1>
          <p className="hero-subtitle">
            Select a season, optionally filter by team, then choose a completed game.
          </p>
        </div>
      </header>

      <main className="layout-grid">
        <section className="panel controls-panel">
          <h2>Choose a game</h2>

          <label className="field-label" htmlFor="season-select">Season</label>
          <select
            id="season-select"
            className="input"
            value={selectedSeason}
            onChange={(e) => setSelectedSeason(e.target.value)}
            disabled={loadingData || seasons.length === 0}
          >
            {seasons.map((season) => (
              <option key={season} value={String(season)}>
                {season}
              </option>
            ))}
          </select>

          <label className="field-label" htmlFor="team-select">Team filter (optional)</label>
          <select
            id="team-select"
            className="input"
            value={selectedTeamId}
            onChange={(e) => setSelectedTeamId(e.target.value)}
            disabled={loadingData || teamOptions.length === 0}
          >
            <option value="">All teams</option>
            {teamOptions.map((team) => (
              <option key={team.id} value={String(team.id)}>
                {team.name}
              </option>
            ))}
          </select>

          <label className="field-label" htmlFor="game-select">Matchup</label>
          <select
            id="game-select"
            className="input"
            value={selectedGameId}
            onChange={(e) => setSelectedGameId(e.target.value)}
            disabled={loadingData || filteredGames.length === 0}
          >
            {filteredGames.map((game) => (
              <option key={game.id} value={game.id}>
                {game.date} — {formatMatchup(game)}
              </option>
            ))}
          </select>

          <button
            className="predict-button"
            onClick={onPredict}
            disabled={!selectedGame || loadingPrediction || loadingData}
          >
            {loadingPrediction ? 'Predicting…' : 'Predict historical result'}
          </button>

          {error && (
            <div className="error-banner">
              <strong>Error:</strong> {error}
            </div>
          )}
        </section>

        {selectedGame && (
          <section className="panel score-panel">
            <div className="score-topline">
              <span className="sport-label">NCAA men&apos;s basketball</span>
              <span className="final-badge">Final</span>
            </div>

            <div className="scoreboard-body">
              <div className="team-block">
                <div className="team-name">{selectedGame.team1Name}</div>
                <div className="team-record">{formatRecord(selectedGame.team1WinsBefore, selectedGame.team1LossesBefore)}</div>
              </div>

              <div className="score-block">
                <span className="score-value">{selectedGame.team1Score}</span>
                <span className="score-separator">-</span>
                <span className="score-value">{selectedGame.team2Score}</span>
              </div>

              <div className="team-block team-block-right">
                <div className="team-name">{selectedGame.team2Name}</div>
                <div className="team-record">{formatRecord(selectedGame.team2WinsBefore, selectedGame.team2LossesBefore)}</div>
              </div>
            </div>

            <div className="scoreboard-footer">
              <span>{new Date(selectedGame.date).toLocaleDateString()}</span>
              <span>{formatMatchup(selectedGame)}</span>
            </div>
          </section>
        )}

        {selectedGame && (
          <section className="panel result-panel">
            <h2>Prediction result</h2>
            <div className="result-grid">
              <div className="result-row">
                <span className="result-label">Matchup</span>
                <span className="result-value">{formatMatchup(selectedGame)}</span>
              </div>
              <div className="result-row">
                <span className="result-label">Actual score</span>
                <span className="result-value">{formatScoreLine(selectedGame)}</span>
              </div>
              <div className="result-row">
                <span className="result-label">Predicted score</span>
                <span className="result-value muted">{formatPredictedScoreLine(selectedGame, prediction)}</span>
              </div>
              <div className="result-row">
                <span className="result-label">Actual winner</span>
                <span className="result-value">{actualWinner(selectedGame)}</span>
              </div>
              <div className="result-row">
                <span className="result-label">Predicted winner</span>
                <span className="result-value muted">{prediction ? predictedWinner(selectedGame, prediction) : '—'}</span>
              </div>
              <div className="result-row">
                <span className="result-label">Winner prediction correct</span>
                <span className="result-value muted">{prediction ? winnerCorrect(selectedGame, prediction) : '—'}</span>
              </div>
              <div className="result-row">
                <span className="result-label">Score</span>
                <span className="result-value muted">{prediction ? rmseScore(selectedGame, prediction) : '—'}</span>
              </div>
            </div>
          </section>
        )}

        <section className="panel log-panel">
          <h2>Activity log</h2>
          <div className="log-list">
            {logs.length === 0 ? (
              <div className="log-empty">No events yet.</div>
            ) : (
              logs.map((entry, index) => (
                <div key={`${entry.timestamp}-${index}`} className={`log-item log-${entry.level}`}>
                  <div className="log-meta">[{entry.timestamp}] {entry.level.toUpperCase()}</div>
                  <div>{entry.message}</div>
                </div>
              ))
            )}
          </div>
        </section>
      </main>
    </div>
  );
}