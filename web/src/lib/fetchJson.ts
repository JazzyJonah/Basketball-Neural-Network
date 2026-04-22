export function assetUrl(path: string): string {
  const clean = path.replace(/^\/+/, '');
  return `${import.meta.env.BASE_URL}${clean}`;
}

export async function fetchJson<T>(path: string): Promise<T> {
  const url = assetUrl(path);
  const response = await fetch(url);
  const text = await response.text();

  if (!response.ok) {
    throw new Error(
      `Failed to fetch ${url}: ${response.status} ${response.statusText}\n` +
      `Response preview: ${text.slice(0, 200)}`
    );
  }

  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error(
      `Failed to parse JSON from ${url}.\n` +
      `Response preview: ${text.slice(0, 200)}`
    );
  }
}