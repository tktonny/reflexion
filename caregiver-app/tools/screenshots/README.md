# Layout screenshots across phone and foldable geometries

Renders every caregiver screen at a dozen device sizes and writes a PNG per (device, screen) so layout
problems can be reviewed side by side instead of found one phone at a time.

```bash
# terminal 1 — the app, on web
npm run web -- --port 8088

# terminal 2
npm i --no-save puppeteer-core
node tools/screenshots/shoot.mjs ../dist-screenshots

# one device or one screen while iterating
ONLY_DEVICE=08-zfold5 ONLY_SCREEN=b-alerts node tools/screenshots/shoot.mjs /tmp/one
```

Output is `dist-screenshots/<device>__<screen>.png`, plus `DEVICES.txt` (the matrix) and `ISSUES.txt` when
anything was detected. The directory is gitignored.

## It serves fixtures, not production

Every request to the production origin is intercepted and answered from `fixtures.mjs`. Nothing reaches the
real backend, no test account is created, and the content is byte-identical in every shot — which is what
makes two screenshots comparable.

The fixtures deliberately carry **worst-case** content, because that is what breaks a layout and average
content hides:

- a long loved-one name (`Grandma Siew Lan Tan`) that wraps a card title
- a long mirror name (`Grandma's bedroom mirror upstairs`)
- the longest real alert body (the mirror-offline explanation)
- a patient with no mirror and a patient missing consent, so both empty/prompt states render
- alerts in every type and both read states

Change the fixtures, not the app, when you want to test a different shape of content.

## Two things that will waste your afternoon

**CORS.** The page is served from `localhost` and the app calls the production origin, so a faked response
without `access-control-allow-origin` is discarded by the browser before the app sees it. On screen that is
"We could not load your alerts" — identical to a real outage, and easy to mistake for a bug in the app. The
interceptor sets CORS headers and answers the `OPTIONS` preflight; a fixture that returns 4xx is reported as
`fixture MISS` for the same reason.

**Expo's dev error toast** is pinned to the bottom of the viewport and covers the tab bar. It is hidden
before each screenshot. It currently reports react-native-web's nested-`<button>` warning, which comes from
nested Pressables and is a web rendering artifact — a phone does not render buttons, so this is not a native
defect.

## What this does and does not tell you

Trustworthy: wrapping, truncation, cramping, horizontal overflow (measured, not eyeballed — `scrollWidth`
against `clientWidth`), whether a control is reachable without scrolling, how a card behaves from 320dp to
744dp.

Not trustworthy: exact typography and shadows (react-native-web differs from native), safe-area insets and
notch/cutout behaviour, OS font-scale accessibility settings (RN reads those from the platform, not from the
browser), and anything involving the camera, secure storage or push.

For those, use a real device. This is for the class of problem where one look at twelve widths answers the
question faster than twelve phones would.
