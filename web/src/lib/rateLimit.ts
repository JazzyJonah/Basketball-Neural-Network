const STORAGE_KEY = 'basketball_predictor_request_timestamps'
const LIMIT = 15
const WINDOW_MS = 60_000

export function checkRateLimit(now = Date.now()): { allowed: boolean; retryAfterMs: number } {
  const raw = window.localStorage.getItem(STORAGE_KEY)
  const timestamps = raw ? (JSON.parse(raw) as number[]) : []
  const recent = timestamps.filter((ts) => now - ts < WINDOW_MS)

  if (recent.length >= LIMIT) {
    const oldest = Math.min(...recent)
    return { allowed: false, retryAfterMs: WINDOW_MS - (now - oldest) }
  }

  recent.push(now)
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(recent))
  return { allowed: true, retryAfterMs: 0 }
}
