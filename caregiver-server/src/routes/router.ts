import { Router } from 'express'
import { signInRouter } from './auth/sign-in.js'
import { addPatientsRouter } from './nurse-patient-config/add-patients.js'
import { createConfigRouter } from './nurse-patient-config/create.js'
import { latestConfigRouter } from './nurse-patient-config/latest.js'
import { mirrorConnectRouter } from './nurse-patient-config/mirrors/connect/index.js'
import { mirrorsRouter } from './nurse-patient-config/mirrors/index.js'
import { notificationsRouter } from './nurse-patient-config/notifications.js'
import { conversationSessionRouter } from './conversation-session.js'
import { conversationSessionCountsRouter } from './conversation-session-counts.js'
import { conversationSessionsByDayRouter } from './conversation-sessions-by-day.js'
import { patientSummaryRouter } from './patient-summary.js'
import { patientTrendRouter } from './patient-trend.js'
import { qwenTokenRouter } from './qwen-token.js'

export const router = Router()

router.use('/auth/sign-in', signInRouter)
router.use('/nurse-patient-config/create', createConfigRouter)
router.use('/nurse-patient-config/add-patients', addPatientsRouter)
router.use('/nurse-patient-config/latest', latestConfigRouter)
router.use('/nurse-patient-config/notifications', notificationsRouter)
router.use('/nurse-patient-config/mirrors', mirrorsRouter)
router.use('/nurse-patient-config/mirrors/connect', mirrorConnectRouter)
router.use('/conversation-session', conversationSessionRouter)
router.use('/conversation-session-counts', conversationSessionCountsRouter)
router.use('/conversation-sessions-by-day', conversationSessionsByDayRouter)
router.use('/patient-summary', patientSummaryRouter)
router.use('/patient-trend', patientTrendRouter)
router.use('/api/qwen-token', qwenTokenRouter)
