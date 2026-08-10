// Ambient home-screen weather (doc: Home shows "weather — icon + temperature only"). Uses open-meteo,
// a public keyless API, for a display-only widget. This is NOT the agent web tool — it needs no
// server allowlist because it's a fixed, read-only forecast endpoint for the paired location. Returns
// null on any failure so the widget simply hides (never blocks the home screen).

export type CurrentWeather = { tempC: number; label: string }
export type WeatherLocation = { city?: string; latitude?: number; longitude?: number }

const WEATHER_LABELS: Record<number, string> = {
  0: 'Clear', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
  45: 'Foggy', 48: 'Foggy', 51: 'Light drizzle', 53: 'Drizzle', 55: 'Drizzle',
  61: 'Light rain', 63: 'Rain', 65: 'Heavy rain', 66: 'Freezing rain', 67: 'Freezing rain',
  71: 'Light snow', 73: 'Snow', 75: 'Heavy snow', 77: 'Snow grains',
  80: 'Showers', 81: 'Showers', 82: 'Heavy showers', 85: 'Snow showers', 86: 'Snow showers',
  95: 'Thunderstorm', 96: 'Thunderstorm', 99: 'Thunderstorm',
}

async function getJson(url: string): Promise<any | null> {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(8000) })
    if (!response.ok) return null
    return await response.json()
  } catch {
    return null
  }
}

export async function fetchCurrentWeather(location: WeatherLocation): Promise<CurrentWeather | null> {
  let latitude = location.latitude
  let longitude = location.longitude
  if ((latitude == null || longitude == null) && location.city) {
    const geo = await getJson(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(location.city)}&count=1`)
    const match = geo?.results?.[0]
    if (match && typeof match.latitude === 'number' && typeof match.longitude === 'number') {
      latitude = match.latitude
      longitude = match.longitude
    }
  }
  if (latitude == null || longitude == null) return null
  const forecast = await getJson(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,weather_code`)
  const current = forecast?.current
  if (!current || typeof current.temperature_2m !== 'number') return null
  return { tempC: Math.round(current.temperature_2m), label: WEATHER_LABELS[current.weather_code] ?? '' }
}
