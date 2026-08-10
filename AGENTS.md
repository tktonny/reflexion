# Reflexion canonical repository instructions

The canonical Reflexion project root is:

`/Users/chloetan/Documents/Reflexion`

Do not edit, build, or release from `/Users/chloetan/Documents/Reflexion app` or `/Users/chloetan/Documents/Reflexion Mirror`.

Before making substantial caregiver or Mirror product changes, read:

- `docs/product-source-of-truth/README.md`
- `docs/product-source-of-truth/SUBSEQUENT_CHANGES.md`
- the relevant architecture documents under `docs/product-source-of-truth/`
- the relevant numbered PNG references under `docs/product-source-of-truth/`

Product priority is:

1. Subsequent explicit product amendments documented in `SUBSEQUENT_CHANGES.md`.
2. The approved architecture and UI references under `docs/product-source-of-truth/`.
3. Non-conflicting implementation work.

Do not restore routes, features, layouts, or behaviours that the approved architecture removes or defers. Do not create duplicate routes for one canonical screen. Keep technical/device state separate from loved-one status, and never expose raw network/provider errors or credentials in consumer UI.
