import { qwenChat } from './qwenClient'
import { getPatientMemory, savePatientMemory } from './sessionSync'
import type { ChatMessage } from '../hooks/conversationTypes'

/**
 * Summarize a finished conversation into a few durable, non-clinical facts and merge them into the
 * patient's backend memory, so Aria remembers the patient across daily chats. Best-effort: any failure
 * (short ticket expired, model error) leaves the prior memory untouched and never blocks the UI.
 */
export async function updatePatientMemoryFromChat(messages: ChatMessage[]): Promise<void> {
  const transcript = messages
    .filter((m) => (m.role === 'user' || m.role === 'assistant') && m.text.trim().length > 0)
    .map((m) => `${m.role === 'user' ? 'Patient' : 'Aria'}: ${m.text.trim()}`)
    .join('\n')
    .slice(0, 4000)
  // Not enough was actually said to be worth remembering.
  if (transcript.replace(/\s/g, '').length < 20) return

  const existing = await getPatientMemory()
  const prompt =
    'You maintain a SHORT memory of durable, non-clinical facts about an elderly person, so a warm ' +
    "daily voice companion can remember them across chats. Given the existing memory and today's " +
    'conversation, output an UPDATED memory of AT MOST 8 concise facts: names, family, pets, interests, ' +
    'routines, food/music preferences, and notable recent life events. Keep facts that are still true, ' +
    'add new ones, and merge duplicates. NEVER include health diagnoses, cognitive/clinical judgments, ' +
    'medication schedules, or any score. Output ONLY the facts, one per line, no numbering or commentary.\n\n' +
    `Existing memory:\n${existing.length ? existing.join('\n') : '(none yet)'}\n\n` +
    `Today's conversation:\n${transcript}`

  const reply = await qwenChat([{ role: 'user', content: prompt }], { maxTokens: 300, temperature: 0.2 })
  const facts = reply
    .split('\n')
    .map((line) => line.replace(/^[-*•\d.)\s]+/, '').trim())
    .filter((line) => line.length > 0)
    .slice(0, 8)
  if (facts.length > 0) await savePatientMemory(facts)
}
