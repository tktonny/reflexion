import { ApiError, badRequest } from '../platform/errors.js'

export async function getWeather(input: { city?: string; latitude?: number; longitude?: number; language?: string }) {
  const location = input.latitude !== undefined && input.longitude !== undefined
    ? { latitude: input.latitude, longitude: input.longitude, name: input.city || 'Configured location', timezone: 'auto' }
    : await geocode(input.city || '', input.language)
  if (Math.abs(location.latitude) > 90 || Math.abs(location.longitude) > 180) throw badRequest('INVALID_LOCATION', 'Latitude or longitude is outside the valid range.')
  const url = new URL('https://api.open-meteo.com/v1/forecast')
  url.search = new URLSearchParams({
    latitude: String(location.latitude), longitude: String(location.longitude), timezone: 'auto', forecast_days: '3',
    current: 'temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m',
    daily: 'weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset',
  }).toString()
  const response = await fetch(url, { signal: AbortSignal.timeout(8000) })
  const body = await response.json().catch(() => null) as Record<string, any> | null
  if (!response.ok || !body?.current) throw new ApiError(502, 'WEATHER_PROVIDER_FAILED', 'Weather service is temporarily unavailable.', true)
  return {
    provider: 'open-meteo', location: { name: location.name, latitude: location.latitude, longitude: location.longitude, timezone: body.timezone },
    current: { observedAt: body.current.time, temperatureC: body.current.temperature_2m, apparentTemperatureC: body.current.apparent_temperature,
      precipitationMm: body.current.precipitation, windSpeedKmh: body.current.wind_speed_10m,
      weatherCode: body.current.weather_code, condition: weatherCondition(body.current.weather_code) },
    daily: Array.isArray(body.daily?.time) ? body.daily.time.map((date: string, index: number) => ({
      date, condition: weatherCondition(body.daily.weather_code?.[index]), weatherCode: body.daily.weather_code?.[index],
      temperatureMaxC: body.daily.temperature_2m_max?.[index], temperatureMinC: body.daily.temperature_2m_min?.[index],
      precipitationProbabilityMax: body.daily.precipitation_probability_max?.[index], sunrise: body.daily.sunrise?.[index], sunset: body.daily.sunset?.[index],
    })) : [], fetchedAt: new Date().toISOString(),
  }
}

export async function webSearch(input: { query: string; language?: string; freshness?: string }) {
  const key = process.env.BRAVE_SEARCH_API_KEY?.trim()
  if (!key) throw new ApiError(503, 'WEB_SEARCH_NOT_CONFIGURED', 'Web search is not configured on this server.', true)
  const url = new URL('https://api.search.brave.com/res/v1/web/search')
  url.search = new URLSearchParams({ q: input.query, count: '5', safesearch: 'strict',
    ...(input.language ? { search_lang: normalizeLanguage(input.language) } : {}),
    ...(input.freshness ? { freshness: input.freshness } : {}),
  }).toString()
  const response = await fetch(url, { headers: { Accept: 'application/json', 'X-Subscription-Token': key }, signal: AbortSignal.timeout(8000) })
  const body = await response.json().catch(() => null) as Record<string, any> | null
  if (!response.ok) throw new ApiError(502, 'WEB_SEARCH_PROVIDER_FAILED', 'Web search is temporarily unavailable.', true)
  return { provider: 'brave', query: input.query, fetchedAt: new Date().toISOString(),
    results: (body?.web?.results || []).slice(0, 5).map((item: Record<string, any>) => ({ title: item.title, url: item.url, description: item.description, age: item.age })) }
}

async function geocode(city: string, language?: string) {
  if (city.trim().length < 2) throw badRequest('LOCATION_REQUIRED', 'A configured city or coordinates are required for weather.')
  const url = new URL('https://geocoding-api.open-meteo.com/v1/search')
  url.search = new URLSearchParams({ name: city.trim(), count: '1', format: 'json', language: normalizeLanguage(language || 'en') }).toString()
  const response = await fetch(url, { signal: AbortSignal.timeout(8000) })
  const body = await response.json().catch(() => null) as { results?: Array<Record<string, any>> } | null
  const result = body?.results?.[0]
  if (!response.ok || !result) throw new ApiError(404, 'LOCATION_NOT_FOUND', 'The requested city could not be found.')
  return { latitude: Number(result.latitude), longitude: Number(result.longitude), name: [result.name, result.admin1, result.country].filter(Boolean).join(', '), timezone: result.timezone }
}

function normalizeLanguage(language: string) { return language.toLowerCase().startsWith('zh') || language === 'mandarin' ? 'zh' : language.slice(0, 2).toLowerCase() }

function weatherCondition(code: number) {
  if (code === 0) return 'clear'
  if ([1, 2, 3].includes(code)) return 'partly_cloudy'
  if ([45, 48].includes(code)) return 'fog'
  if (code >= 51 && code <= 67) return 'rain'
  if (code >= 71 && code <= 77) return 'snow'
  if (code >= 80 && code <= 82) return 'rain_showers'
  if (code >= 85 && code <= 86) return 'snow_showers'
  if (code >= 95) return 'thunderstorm'
  return 'unknown'
}
