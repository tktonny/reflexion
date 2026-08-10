# Daily Conversation v2

## Product contract

The daily check-in lasts about 3–5 minutes and must feel like a kind friend checking in, never a test. Patient speech is the captured clinical input; Aria does not diagnose, score, or describe cognitive signals during the live conversation.

| Order | Stage | Target | Aria prompt | Captured signal tags |
| --- | --- | ---: | --- | --- |
| 1 | Warm-up | 30s | “Good morning [Name], it's lovely to see you. How are you feeling today?” | `mood`, `speech_initiation`, `response_latency` |
| 2 | Yesterday recall | 60s | Dinner yesterday; sleep last night | `episodic_memory`, `temporal_orientation`, `narrative_coherence` |
| 3 | Present/planning | 60s | Plans today; visitors this week | `executive_function`, `prospective_memory`, `social_connectedness` |
| 4 | Medication reminder | Conditional | Ask only about a currently due caregiver/provider-configured occurrence | `memory`, `caregiver_adjunct` |
| 5 | Reminiscence | 60–90s, twice weekly | Loved holiday or favourite childhood food | `semantic_memory`, `language_richness`, `lexical_diversity`, `speech_fluency` |
| 6 | Close | 30s | Warm thanks, good-morning wish, and goodbye | session completion |

## Deterministic runtime rules

- The base flow contains five accepted patient turns: warm-up, two yesterday turns, and two planning/social turns.
- Reminiscence defaults to Tuesday and Friday. Tuesday uses the holiday prompt; Friday uses childhood food. This can later move to a care-plan field without changing the transport contract.
- A medication turn is included only when `/api/v1/patients/:patientId/reminder-occurrences` returns an unresolved medication occurrence due within 15 minutes or overdue by no more than six hours.
- If medication context cannot be loaded, the stage is omitted. Aria never guesses a medicine, dose, instruction, or schedule.
- The next question and terminal close are selected from accepted patient-turn count and the frozen daily plan. No fixed UI timer advances the conversation.
- After the final enabled patient response, Aria uses a deterministic positive close and does not open a new topic.
- “Remember”, right/wrong, assessment, score, diagnosis, and dementia framing are prohibited in live speech.

## Uploaded evidence

Each patient transcript event carries:

- `protocolVersion: daily-conversation-v2`
- `protocolStage`
- allowlisted `cognitiveSignals`

The caregiver server stores these fields on normalized transcript turns. They are evidence labels for downstream quality and feature extraction, not a live diagnosis and not a direct caregiver red/yellow/green input.
