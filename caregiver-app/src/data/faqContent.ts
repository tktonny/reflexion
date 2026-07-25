// Product FAQ content shown on app/faq.tsx. Real copy, not fixtures — it used to live in mockData.ts
// alongside demo data and the app's shared type definitions, which is how a three-state legacy `Status`
// ended up as the type production screens compiled against.

export const FAQ_ITEMS = [
  {
    q: 'How does Reflexion work?',
    a: 'Reflexion uses a smart mirror with a built-in voice AI companion called Aria. Each morning, Aria has a gentle 5-minute conversation with your loved one. The app shows you a daily summary and alerts you to any changes in their routine.',
  },
  {
    q: 'Does my parent need to do anything to start?',
    a: 'No. Aria greets them automatically each morning at their usual wake time. They just need to talk back. There are no buttons to press and no apps to open.',
  },
  {
    q: 'What does the status colour mean?',
    a: 'Learning routine means Aria is still getting to know their usual days. Doing well means the day followed their usual pattern. Worth checking means something was a little different, such as a shorter or later conversation. Needs attention means it may be worth giving them a call. Each one comes with a short line telling you why.',
  },
  {
    q: 'Will the conversation be stored?',
    a: 'Yes. A summary of each conversation is stored securely so the app can show you how the days compare. You can turn this off under Store session summaries in Settings.',
  },
  {
    q: 'Can I tell Aria about my loved one?',
    a: 'Yes. In Settings, open their profile to set their preferred language, usual wake time, the topics they enjoy, and any speech or hearing difficulties. Aria uses all of it to make the conversation easier for them.',
  },
  {
    q: 'How do I link the mirror to the app?',
    a: 'During onboarding, scan the QR code on the back of the Reflexion device. You can also do this later in Settings.',
  },
  {
    q: 'What languages does Aria speak?',
    a: 'Aria currently supports English and Mandarin. Malay and Tamil support are coming in Phase 2.',
  },
  {
    q: 'Can multiple caregivers monitor the same person?',
    a: 'Not yet — this is planned for a future update. Currently, one caregiver account is linked per loved one.',
  },
];
