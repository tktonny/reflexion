import { MongoClient } from 'mongodb'
import { getMongoUri } from './env.js'

export async function withMongo<T>(callback: (client: MongoClient) => Promise<T>) {
  const client = new MongoClient(getMongoUri())
  await client.connect()
  try {
    return await callback(client)
  } finally {
    await client.close()
  }
}
