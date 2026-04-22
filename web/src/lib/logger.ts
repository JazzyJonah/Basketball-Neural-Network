import { LogEntry } from '../types'

export function makeLogEntry(level: LogEntry['level'], message: string): LogEntry {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    level,
    message,
    timestamp: new Date().toISOString(),
  }
}

export function writeConsoleLog(level: LogEntry['level'], message: string): void {
  const prefix = `[${level.toUpperCase()}]`
  if (level === 'error') console.error(prefix, message)
  else if (level === 'warn') console.warn(prefix, message)
  else console.info(prefix, message)
}
