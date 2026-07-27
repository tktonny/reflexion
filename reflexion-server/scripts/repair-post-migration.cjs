#!/usr/bin/env node
/*
 * Three one-off repairs, in the order they should run. Run from reflexion-server:
 *
 *   node --env-file=.env /root/cleanup-all.cjs            # DRY RUN: prints what it would do, writes nothing
 *   node --env-file=.env /root/cleanup-all.cjs --apply    # actually writes
 *
 * 1. Re-run the monitoring pipeline on daily check-ins the OLD quality thresholds rejected.
 * 2. Record the days a loved one could not check in because our own consent gate refused them, so the
 *    3-day-missed alarm stops blaming the person for a backend defect.
 * 3. Delete the qa-regression-* tenants an end-to-end test left behind.
 *
 * PREREQUISITE for step 1: the new thresholds must already be deployed (PR #31 — floor/ideal quality gate).
 * Re-running against the old code would reject the same sessions all over again.
 */
const { MongoClient } = require('mongodb')

const APPLY = process.argv.includes('--apply')
const uri = process.env.MONGODB_URI
if (!uri) {
  console.error('MONGODB_URI is not set. Run with: node --env-file=.env <this file>')
  process.exit(1)
}

const CHECKIN_CONSENT_PURPOSE = 'home_cognitive_monitoring'
const heading = (text) => console.log(`\n${text}\n${'-'.repeat(text.length)}`)
const iso = (d) => (d ? new Date(d).toISOString().replace('T', ' ').slice(0, 19) : '-')
let idCounter = 0
// Mirrors v1/platform/ids.ts closely enough for one-off rows; the prefix is what downstream code reads.
const newId = (prefix) => `${prefix}_${Date.now().toString(16)}${(idCounter++).toString(16).padStart(4, '0')}${Math.random().toString(16).slice(2, 10)}`

/** The Singapore-local calendar day for an instant, as YYYY-MM-DD. */
function sgDayKey(date) {
  return new Date(new Date(date).getTime() + 8 * 3_600_000).toISOString().slice(0, 10)
}

function addDays(dayKey, days) {
  const base = new Date(`${dayKey}T00:00:00Z`)
  return new Date(base.getTime() + days * 86_400_000).toISOString().slice(0, 10)
}

async function main() {
  const client = new MongoClient(uri)
  await client.connect()
  const db = client.db(process.env.MONGODB_DB || 'ref')
  console.log(`database: ${db.databaseName}   mode: ${APPLY ? 'APPLY (writing)' : 'DRY RUN (no writes)'}`)

  // ── 1. Reprocess the check-ins the old quality gate discarded ───────────────────────────────────
  heading('1. Re-run the pipeline on check-ins the old thresholds rejected')
  const excluded = await db.collection('sessions').find({
    type: 'daily_checkin', state: 'excluded',
  }, { projection: { patientId: 1, createdAt: 1, processingSummary: 1, latestProcessingRevision: 1 } })
    .sort({ createdAt: 1 }).toArray()

  console.log(`excluded daily check-ins: ${excluded.length}`)
  for (const session of excluded) {
    const flags = session.processingSummary?.qualityFlags || []
    console.log(`  ${iso(session.createdAt)}  ${session._id}  patient=${session.patientId}  flags=[${flags.join(',')}]`)
  }
  if (!excluded.length) {
    console.log('  nothing to do')
  } else if (APPLY) {
    // Back to `ingesting`, the state a completed session sits in before the worker picks it up. The pipeline
    // refuses to reprocess anything already completed/excluded/review_pending, which is why the state has to
    // move first. Every write it then makes is an upsert keyed by a fresh revision, so this is safe to repeat.
    for (const session of excluded) {
      await db.collection('sessions').updateOne({ _id: session._id, state: 'excluded' }, {
        $set: { state: 'ingesting', updatedAt: new Date() }, $inc: { stateVersion: 1 },
      })
      await db.collection('outbox_events').insertOne({
        _id: newId('evt'), eventType: 'session.completed', eventVersion: 1, occurredAt: new Date(),
        tenantId: String(session.tenantId || ''), patientId: String(session.patientId),
        aggregateType: 'session', aggregateId: String(session._id),
        correlationId: `requality-sweep:${session._id}`, causationId: undefined,
        payload: { reprocessedBy: 'quality-threshold-change' },
        state: 'pending', attempt: 0, nextAttemptAt: new Date(), createdAt: new Date(),
      })
    }
    console.log(`  requeued ${excluded.length} session(s) — the worker picks them up on its next poll`)
    console.log('  NOTE: eventConsumptions is keyed {eventId, consumerName}, and these are new event ids,')
    console.log('        so the worker will not treat them as already consumed.')
  }

  // ── 2. Stop blaming a loved one for our consent gate ────────────────────────────────────────────
  heading('2. Record the days our consent gate refused, so the missed-streak alarm is honest')
  // The alarm is computed live: monitoring.ts derives missedStreak from the gap since the last completed
  // check-in and SUBTRACTS any away days in that window. Editing daily_statuses would be undone by the next
  // 7pm finalise; an away period is the mechanism the engine already respects.
  const blocked = []
  const patients = await db.collection('patients').find({}, { projection: { displayName: 1, tenantId: 1, timezone: 1 } }).toArray()
  for (const patient of patients) {
    const today = await db.collection('daily_statuses').findOne(
      { patientId: patient._id }, { sort: { localDate: -1 } })
    if (!today || today.finalStatus !== 'red' || today.primaryReason !== 'CHECKIN_MISSED_3_DAYS') continue

    const consent = await db.collection('consents').findOne({
      patientId: patient._id, purpose: CHECKIN_CONSENT_PURPOSE, status: 'granted',
    }, { sort: { signedAt: 1 } })
    if (!consent) { console.log(`  ${patient._id} is red but still has NO consent — fix that first`); continue }

    const consentDay = sgDayKey(consent.signedAt || consent.createdAt)
    const streak = Number(today.missedStreakAfterToday || 0)
    const startsOn = addDays(consentDay, -Math.max(streak, 1))
    // Up to the day before consent existed: from the consent day onward a check-in was genuinely possible.
    const endsOn = addDays(consentDay, -1)
    blocked.push({ patient, startsOn, endsOn, consentDay, streak, backfilled: consent.documentVersion })
  }

  if (!blocked.length) {
    console.log('  no patient is showing a red missed-streak — nothing to do')
  }
  for (const entry of blocked) {
    console.log(`  ${entry.patient._id} (${entry.patient.displayName || '?'})  streak=${entry.streak}`);
    console.log(`     consent granted on ${entry.consentDay} (documentVersion=${entry.backfilled})`)
    console.log(`     would mark away: ${entry.startsOn} .. ${entry.endsOn}`)
    if (!APPLY) continue
    const existing = await db.collection('away_periods').findOne({
      patientId: entry.patient._id, startsOn: entry.startsOn, endsOn: entry.endsOn, state: 'active',
    })
    if (existing) { console.log('     already recorded'); continue }
    await db.collection('away_periods').insertOne({
      _id: newId('away'), tenantId: entry.patient.tenantId, patientId: entry.patient._id,
      startsOn: entry.startsOn, endsOn: entry.endsOn,
      timezone: entry.patient.timezone || 'Asia/Singapore',
      // On the record deliberately: this was our defect, not an absence, and a future reader should be able
      // to tell the difference between "they were away" and "we would not let them check in".
      reason: 'Daily check-ins were refused by a backend consent gate during these days; no consent record existed yet.',
      createdBy: 'system:consent-gate-repair', state: 'active', createdAt: new Date(),
    })
    console.log('     recorded')
  }
  if (blocked.length && APPLY) {
    console.log('  the red clears on the next finalise pass (reflexion-finalize runs every 60s)')
  }

  // ── 3. Remove the end-to-end test tenants ──────────────────────────────────────────────────────
  heading('3. Delete the qa-regression-* tenants')
  const qaUsers = await db.collection('users').find({ emailNormalized: /^qa-regression-/ },
    { projection: { tenantId: 1, emailNormalized: 1 } }).toArray()
  const qaTenants = [...new Set(qaUsers.map((user) => user.tenantId).filter(Boolean))]
  console.log(`qa users: ${qaUsers.length}  tenants: ${qaTenants.length}`)
  for (const user of qaUsers) console.log(`  ${user.emailNormalized}  tenant=${user.tenantId}`)

  if (!qaTenants.length) {
    console.log('  nothing to do')
  } else if (APPLY) {
    const scoped = ['users', 'patients', 'care_relationships', 'consents', 'care_plans', 'auth_sessions',
      'notification_devices', 'idempotency_records', 'audit_events', 'daily_statuses',
      'operational_baselines', 'notifications', 'sessions', 'away_periods', 'feedback']
    for (const name of scoped) {
      const result = await db.collection(name).deleteMany({ tenantId: { $in: qaTenants } })
      if (result.deletedCount) console.log(`  ${name.padEnd(22)} -${result.deletedCount}`)
    }
    const tenants = await db.collection('tenants').deleteMany({ _id: { $in: qaTenants } })
    if (tenants.deletedCount) console.log(`  ${'tenants'.padEnd(22)} -${tenants.deletedCount}`)
  }

  console.log(APPLY
    ? '\nDone. Re-run scripts/diagnose-v1-migration-state.cjs to confirm, then check the trend chart.'
    : '\nDRY RUN — nothing was written. Re-run with --apply to make these changes.')
  await client.close()
}

main().catch((error) => { console.error('FAILED:', error.message); process.exit(1) })
