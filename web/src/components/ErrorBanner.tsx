export function ErrorBanner({ message }: { message: string | null }) {
  if (!message) return null
  return (
    <div style={{ background: '#fee2e2', color: '#991b1b', padding: 12, borderRadius: 12, marginBottom: 16 }}>
      <strong>Error:</strong> {message}
    </div>
  )
}
