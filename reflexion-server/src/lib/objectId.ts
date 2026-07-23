import { ObjectId } from 'mongodb'

export function objectIdFromUnknown(value: unknown) {
  if (value instanceof ObjectId) return value
  if (typeof value === 'string' && ObjectId.isValid(value)) return new ObjectId(value)
  if (value && typeof value === 'object' && '$oid' in value && ObjectId.isValid(String((value as { $oid?: string }).$oid))) {
    return new ObjectId(String((value as { $oid?: string }).$oid))
  }
  return null
}

export function objectIdToString(value: unknown) {
  if (value instanceof ObjectId) return value.toHexString()
  if (typeof value === 'string') return value
  if (value && typeof value === 'object' && '$oid' in value) {
    return String((value as { $oid?: string }).$oid || '')
  }
  return ''
}

export function uniqueObjectIds(values: Array<ObjectId | null>) {
  const byId = new Map<string, ObjectId>()
  for (const value of values) {
    if (value) byId.set(value.toHexString(), value)
  }
  return Array.from(byId.values())
}

export function uniqueStrings(values: string[]) {
  return Array.from(new Set(values.filter(Boolean)))
}
