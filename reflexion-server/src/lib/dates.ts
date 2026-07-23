import { TIME_ZONE } from './constants.js'

export function getSingaporeDateKey(date: Date) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    day: '2-digit',
    month: '2-digit',
    timeZone: TIME_ZONE,
    year: 'numeric',
  }).formatToParts(date)
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]))
  return `${values.year}-${values.month}-${values.day}`
}

export function getSingaporeMonthKey(date: Date) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    month: '2-digit',
    timeZone: TIME_ZONE,
    year: 'numeric',
  }).formatToParts(date)
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]))
  return `${values.year}-${values.month}`
}

export function getSingaporeDayBounds(date: Date) {
  const parts = new Intl.DateTimeFormat('en-SG', {
    day: '2-digit',
    month: '2-digit',
    timeZone: TIME_ZONE,
    year: 'numeric',
  }).formatToParts(date)
  const year = Number(parts.find((part) => part.type === 'year')?.value || 0)
  const month = Number(parts.find((part) => part.type === 'month')?.value || 1)
  const day = Number(parts.find((part) => part.type === 'day')?.value || 1)

  return {
    start: new Date(Date.UTC(year, month - 1, day) - 8 * 60 * 60 * 1000),
    end: new Date(Date.UTC(year, month - 1, day + 1) - 8 * 60 * 60 * 1000),
  }
}

export function getSingaporeDayBoundsFromKey(dateKey: string) {
  const [year, month, day] = dateKey.split('-').map(Number)

  return {
    start: new Date(Date.UTC(year, month - 1, day) - 8 * 60 * 60 * 1000),
    end: new Date(Date.UTC(year, month - 1, day + 1) - 8 * 60 * 60 * 1000),
  }
}

export function getSingaporeMonthBounds(monthKey: string) {
  const [year, month] = monthKey.split('-').map(Number)
  const daysInMonth = new Date(year, month, 0).getDate()

  return {
    daysInMonth,
    start: new Date(Date.UTC(year, month - 1, 1) - 8 * 60 * 60 * 1000),
    end: new Date(Date.UTC(year, month, 1) - 8 * 60 * 60 * 1000),
  }
}

export function getSingaporeDayOfMonth(date: Date) {
  const parts = new Intl.DateTimeFormat('en-SG', {
    day: '2-digit',
    timeZone: TIME_ZONE,
  }).formatToParts(date)

  return Number(parts.find((part) => part.type === 'day')?.value || 0)
}

export function getMissedDays(createdAt: Date | null) {
  if (!createdAt) return 999

  const today = getSingaporeDaySerial(new Date())
  const spokenDay = getSingaporeDaySerial(createdAt)
  return Math.max(0, today - spokenDay)
}

function getSingaporeDaySerial(date: Date) {
  const parts = new Intl.DateTimeFormat('en-SG', {
    day: '2-digit',
    month: '2-digit',
    timeZone: TIME_ZONE,
    year: 'numeric',
  }).formatToParts(date)
  const year = Number(parts.find((part) => part.type === 'year')?.value || 0)
  const month = Number(parts.find((part) => part.type === 'month')?.value || 1)
  const day = Number(parts.find((part) => part.type === 'day')?.value || 1)
  return Math.floor(Date.UTC(year, month - 1, day) / 86400000)
}
