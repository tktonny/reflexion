# Reflexion product source of truth

These files define the approved Reflexion product baseline. Subsequent explicit product amendments may override specific details. Do not replace them with older repository documents.

## Priority

1. Subsequent explicit product amendments documented in `SUBSEQUENT_CHANGES.md`.
2. The architecture documents and numbered UI references in this directory.
3. Non-conflicting implementation work.

## Reference layout

- `caregiver/architecture/` — approved caregiver product and functional architecture.
- `caregiver/ui/` — approved caregiver screen references, identified by `AUTH-`, `SETUP-`, `HOME-`, `LOVED-`, `ACT-`, `CHAT-`, `SET-`, and `RES-` IDs.
- `mirror/architecture/` — approved Mirror state/conversation architecture. The later `UPDATED_voice_reply_end_conversation` DOCX records the voice-reply and spoken-Goodbye amendment.
- `mirror/ui/` — approved Mirror state references, identified by `MIR-` IDs.

## Verification records

- `docs/migration/final-product-source-inventory.md` inventories every retained architecture document and UI PNG.
- `docs/migration/caregiver-screen-matrix.md` maps every caregiver reference to both existing implementations.
- `docs/migration/mirror-screen-matrix.md` maps every Mirror reference to both existing implementations.

The matrices distinguish implementation candidates from architecture compliance and visual compliance. A screen is not a compliance pass merely because a similarly named route exists.

The original source folder remains authoritative for provenance and is intentionally not modified:

`/Users/chloetan/Downloads/Reflexion FINAL architectures and UI`
