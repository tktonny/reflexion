// Seed a NursePatientConfig (one caregiver + one patient) with FIXED ObjectIds into DB 'ref',
// so check-ins saved with the demo IDs (src/config/conversationMode.ts DEMO_IDS) show up in the
// caregiver app / server. Run against YOUR Atlas:
//   MONGODB_URI="mongodb+srv://..." node --env-file=.env.server.local server/seed-demo-patient.mjs
// (or put MONGODB_URI in .env.server.local). Idempotent (upsert).

import { MongoClient, ObjectId } from 'mongodb'

const uri = process.env.MONGODB_URI
if (!uri) { console.error('Set MONGODB_URI in the environment or .env.server.local.'); process.exit(1) }

const NURSE = process.env.EXPO_PUBLIC_DEMO_NURSE_ID || '64f0000000000000000000a1'
const PATIENT = process.env.EXPO_PUBLIC_DEMO_PATIENT_ID || '65f0000000000000000000b2'

const client = new MongoClient(uri)
await client.connect()
try {
  const db = client.db('ref')
  const now = new Date()
  await db.collection('NursePatientConfig').updateOne(
    { _id: new ObjectId(NURSE) },
    {
      $set: {
        name: 'Demo Caregiver',
        email: 'demo-caregiver@reflexion.test',
        passwordHash: '',
        phoneNumber: '',
        relationshipToElderly: 'other',
        pushNotificationsEnabled: true,
        alertSensitivity: 'only_important_changes',
        preferredDailySummaryTime: '09:00',
        updatedAt: now,
        patients: [
          {
            _id: new ObjectId(PATIENT),
            name: 'Demo Patient',
            age: 78,
            gender: 'other',
            preferredLanguage: 'english',
            usualWakeTime: '08:00',
            keyTopics: ['family'],
            mirrorId: null,
            mirrorName: 'Demo Mirror',
            mirrorVerified: false,
            mirrorPairingStatus: '',
            timezone: 'Asia/Singapore',
          },
        ],
      },
      $setOnInsert: { createdAt: now },
    },
    { upsert: true },
  )
  const count = await db.collection('Conversation').countDocuments({ patientId: new ObjectId(PATIENT) })
  console.log(`Seeded NursePatientConfig  nurseId=${NURSE}  patientId=${PATIENT}`)
  console.log(`Existing Conversations for this patient: ${count}`)
  console.log('Caregiver app: sign in as this nurse (or query nurseId) → this patient should appear.')
} finally {
  await client.close()
}
