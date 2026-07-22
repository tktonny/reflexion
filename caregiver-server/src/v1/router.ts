import { Router } from 'express'
import { carePlanRouter } from './routes/carePlan.js'
import { devicesRouter } from './routes/devices.js'
import { identityRouter } from './routes/identity.js'
import { monitoringRouter } from './routes/monitoring.js'
import { notificationsRouter } from './routes/notifications.js'
import { patientsRouter } from './routes/patients.js'
import { sessionsRouter } from './routes/sessions.js'
import { toolsRouter } from './routes/tools.js'

export const v1Router = Router()

v1Router.use(identityRouter)
v1Router.use(patientsRouter)
v1Router.use(devicesRouter)
v1Router.use(sessionsRouter)
v1Router.use(toolsRouter)
v1Router.use(carePlanRouter)
v1Router.use(monitoringRouter)
v1Router.use(notificationsRouter)
