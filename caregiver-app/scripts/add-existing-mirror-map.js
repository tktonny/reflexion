const fs = require('fs');
const path = require('path');
const { MongoClient, ObjectId } = require('mongodb');

const DB_NAME = 'ref';
const COLLECTION_NAME = 'MirrorIdToNurseIdMap';

const mapping = {
  nurseId: new ObjectId('6a2ef45e5f82314f47f408b4'),
  patientId: new ObjectId('6a2ef45d5f82314f47f408b3'),
  mirrorId: new ObjectId('6a2ef45d5f82314f47f408b2'),
};

function loadEnvValue(key) {
  const envPath = path.join(process.cwd(), '.env');
  if (!fs.existsSync(envPath)) {
    return process.env[key];
  }

  const line = fs
    .readFileSync(envPath, 'utf8')
    .split(/\r?\n/)
    .find((entry) => entry.startsWith(`${key}=`));

  return line ? line.slice(key.length + 1).trim() : process.env[key];
}

async function main() {
  const uri = loadEnvValue('MONGODB_URI');
  if (!uri) {
    throw new Error('MONGODB_URI is not set in .env or process.env');
  }

  const client = new MongoClient(uri);
  await client.connect();

  try {
    const now = new Date();
    const result = await client
      .db(DB_NAME)
      .collection(COLLECTION_NAME)
      .updateMany(
        {
          $or: [
            { mirrorId: mapping.mirrorId },
            { mirrorId: mapping.mirrorId.toHexString() },
          ],
        },
        {
          $set: {
            nurseId: mapping.nurseId,
            patientId: mapping.patientId,
            updatedAt: now,
          },
          $unset: {
            mirrorName: '',
            patientName: '',
          },
        },
      );

    console.log({
      matchedCount: result.matchedCount,
      modifiedCount: result.modifiedCount,
    });
  } finally {
    await client.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
