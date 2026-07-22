import { type ClientSession, type Db, MongoClient } from 'mongodb'
import { getMongoUri } from './env.js'

let clientPromise: Promise<MongoClient> | undefined

export function getMongoClient() {
  if (!clientPromise) {
    const client = new MongoClient(getMongoUri(), {
      maxPoolSize: Number(process.env.MONGODB_MAX_POOL_SIZE || 20),
      minPoolSize: Number(process.env.MONGODB_MIN_POOL_SIZE || 0),
      retryReads: true,
      retryWrites: true,
    })
    clientPromise = client.connect().catch((error) => {
      clientPromise = undefined
      throw error
    })
  }
  return clientPromise
}

export async function getDb() {
  const client = await getMongoClient()
  return client.db(process.env.MONGODB_DB || 'ref')
}

export async function withMongo<T>(callback: (client: MongoClient) => Promise<T>) {
  return callback(await getMongoClient())
}

export async function inTransaction<T>(callback: (db: Db, session: ClientSession) => Promise<T>) {
  const client = await getMongoClient()
  return client.withSession(async (session) => session.withTransaction(
    () => callback(client.db(process.env.MONGODB_DB || 'ref'), session),
    {
      readConcern: { level: 'snapshot' },
      writeConcern: { w: 'majority' },
      readPreference: 'primary',
    },
  ))
}

export async function closeMongo() {
  const current = clientPromise
  clientPromise = undefined
  if (current) await (await current).close()
}
