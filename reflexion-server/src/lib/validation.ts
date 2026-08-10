export const RELATIONSHIPS = ['parent', 'sibling', 'spouse', 'inlaw', 'grandpa', 'grandma', 'other'] as const
export const GENDERS = ['male', 'female', 'other'] as const
export const LANGUAGES = ['english', 'mandarin', 'other'] as const
export const TOPICS = ['family', 'food', 'travel', 'work', 'others'] as const
export const ALERT_SENSITIVITIES = [
  'notify_me_about_everything',
  'only_important_changes',
  'only_urgent_alerts',
] as const
export const SUMMARY_TIMES = ['09:00', '19:00'] as const

export function isOneOf<T extends readonly string[]>(value: unknown, options: T): value is T[number] {
  return typeof value === 'string' && options.includes(value)
}
