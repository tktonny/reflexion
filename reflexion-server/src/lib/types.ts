import type { ObjectId } from 'mongodb'

export type ConversationLog = {
  sentence?: string
  role?: string
  words?: number
  duration?: number
  wordsPerSecond?: number
}

export type Conversation = {
  _id?: ObjectId
  conversationId?: ObjectId
  duration?: number
  words?: number
  exchanges?: number
  avgLatency?: number
  logs?: ConversationLog[]
  sessionStatus?: string
  createdAt?: Date
  updatedAt?: Date
}

export type DailyPatientStatus = {
  _id?: ObjectId
  patientId: ObjectId
  nurseId?: ObjectId
  date: string
  timezone: string
  preferredDailySummaryTime: string
  status: 'green' | 'yellow' | 'red' | null
  missed: boolean
  sessionCount: number
  completedSessionCount: number
  createdAt?: Date
  updatedAt?: Date
}

export type ConversationMap = {
  conversationId?: ObjectId
  patientId?: ObjectId
  nurseId?: ObjectId
  createdAt?: Date
  updatedAt?: Date
}

export type StoredPatient = {
  _id?: ObjectId
  name?: string
  phoneNumber?: string
  age?: number
  gender?: string
  preferredLanguage?: string
  usualWakeTime?: string
  speechOrHearingConditions?: string | null
  speechSpeed?: string
  mirrorName?: string
  photoUrl?: string
  keyTopics?: string[]
  keyTopicsOtherText?: string | null
}
