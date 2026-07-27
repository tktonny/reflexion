/*
 * Screenshots the caregiver app across mainstream phone and foldable geometries.
 *
 * Runs the Expo web build in the system Chrome and intercepts every call to the production origin, serving
 * fixtures instead — so nothing touches the real backend, the content is identical in every shot, and the
 * fixtures deliberately carry worst-case strings (a long name, a long mirror name, a long technical
 * explanation) because that is what actually breaks a layout.
 *
 *   node shoot.mjs [outputDir]
 */
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import puppeteer from 'puppeteer-core'
import { AUTH_SESSION, PATIENT_A, V1_SESSION, respondFor } from './fixtures.mjs'

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const BASE = 'http://localhost:8088'
const API_HOST = 'reflexion.production.tktonny.top'
const OUT = process.argv[2] || path.join(process.cwd(), 'out')

// Logical (CSS) pixels — the units layout actually works in. dpr only affects raster sharpness.
const DEVICES = [
  { id: '01-iphone-se',        label: 'iPhone SE (2/3rd gen)',        w: 375,  h: 667,  dpr: 2 },
  { id: '02-galaxy-s23',       label: 'Galaxy S23 / narrow Android',  w: 360,  h: 780,  dpr: 3 },
  { id: '03-huawei-p20-pro',   label: 'Huawei P20 Pro (your device)', w: 360,  h: 748,  dpr: 3 },
  { id: '04-iphone-13-15',     label: 'iPhone 13/14/15',              w: 390,  h: 844,  dpr: 3 },
  { id: '05-pixel-7',          label: 'Pixel 7',                      w: 412,  h: 915,  dpr: 2.6 },
  { id: '06-iphone-15-pro-max',label: 'iPhone 15 Pro Max',            w: 430,  h: 932,  dpr: 3 },
  { id: '07-zflip5-open',      label: 'Galaxy Z Flip5 unfolded (tall)', w: 412, h: 1004, dpr: 2.6 },
  { id: '08-zfold5-cover',     label: 'Galaxy Z Fold5 cover (narrowest)', w: 344, h: 882, dpr: 2.6 },
  { id: '09-zfold5-open',      label: 'Galaxy Z Fold5 unfolded (wide)', w: 673, h: 841, dpr: 2.6 },
  { id: '10-pixel-fold-open',  label: 'Pixel Fold unfolded',          w: 701,  h: 841,  dpr: 2.2 },
  { id: '11-ipad-mini',        label: 'iPad mini (upper bound)',      w: 744,  h: 1133, dpr: 2 },
  { id: '12-tiny-legacy',      label: 'Legacy small Android 320dp',   w: 320,  h: 640,  dpr: 2 },
]

const SCREENS = [
  { id: 'a-home',            route: '/(tabs)',                                  wait: 'Good' },
  { id: 'b-alerts',          route: '/(tabs)/alerts',                           wait: 'Alerts' },
  { id: 'c-settings',        route: '/(tabs)/settings',                          wait: 'Settings' },
  { id: 'd-profile',         route: `/profile/${PATIENT_A}`,                     wait: null },
  { id: 'e-mirrors',         route: '/mirror-management',                        wait: null },
  { id: 'f-session-history', route: `/session-history/${PATIENT_A}`,             wait: null },
  { id: 'g-day-detail',      route: `/session-history/${PATIENT_A}/2026-07-27`,  wait: null },
  { id: 'h-trend',           route: `/trend/${PATIENT_A}`,                       wait: null },
  { id: 'c2-set-account',    route: '/settings/account',                        wait: null },
  { id: 'c3-set-notifs',     route: '/settings/notifications',                  wait: null },
  { id: 'c4-set-loved',      route: '/settings/loved-ones',                     wait: null },
  { id: 'c5-set-privacy',    route: '/settings/privacy',                        wait: null },
  { id: 'c6-set-support',    route: '/settings/support',                        wait: null },
  { id: 'i-sign-in',         route: '/sign-in',                                 wait: 'Sign in', anonymous: true },
  { id: 'j-onboarding',      route: '/onboarding',                               wait: null, anonymous: true },
]

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

async function main() {
  await mkdir(OUT, { recursive: true })
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: 'new',
    args: ['--hide-scrollbars', '--no-first-run', '--disable-features=Translate'],
  })

  const only = process.env.ONLY_DEVICE
const onlyScreen = process.env.ONLY_SCREEN
const report = []
  for (const device of (only ? DEVICES.filter((d) => d.id.includes(only)) : DEVICES)) {
    for (const screen of (onlyScreen ? SCREENS.filter((s) => s.id.includes(onlyScreen)) : SCREENS)) {
      const page = await browser.newPage()
      await page.setViewport({ width: device.w, height: device.h, deviceScaleFactor: device.dpr, isMobile: true, hasTouch: true })
      await page.setRequestInterception(true)
      // The page is served from localhost while the app calls the production origin, so every faked response
      // needs CORS headers or the browser discards it before the app ever sees it — which renders as
      // "We could not load your alerts", indistinguishable on screen from a real outage.
      const cors = {
        'access-control-allow-origin': '*',
        'access-control-allow-methods': 'GET,POST,PATCH,PUT,DELETE,OPTIONS',
        'access-control-allow-headers': 'authorization,content-type,idempotency-key,if-match,accept',
        'access-control-max-age': '600',
      }
      page.on('request', (request) => {
        const url = request.url()
        if (!url.includes(API_HOST)) return void request.continue()
        if (request.method() === 'OPTIONS') return void request.respond({ status: 204, headers: cors, body: '' })
        const marker = '/api/v1'
        const index = url.indexOf(marker)
        const body = index === -1 ? null : respondFor(url.slice(index + marker.length))
        if (!body) {
          return void request.respond({ status: 404, headers: cors, contentType: 'application/json',
            body: JSON.stringify({ error: { code: 'NOT_FOUND', message: 'no fixture' } }) })
        }
        request.respond({ status: 200, headers: cors, contentType: 'application/json', body: JSON.stringify(body) })
      })
      // A missing fixture looks exactly like a layout problem in a screenshot, so it gets reported loudly.
      page.on('response', (response) => {
        if (response.url().includes(API_HOST) && response.status() >= 400) {
          report.push(`  ! ${device.id}/${screen.id} fixture MISS ${response.status()} ${response.url().split('/api/v1')[1]}`)
        }
      })
      page.on('pageerror', (error) => report.push(`  ! ${device.id}/${screen.id} pageerror: ${String(error.message).slice(0, 120)}`))

      if (screen.anonymous) {
        // Pages share one browser context, so a session written for an earlier screen is still in
        // localStorage — which sent /sign-in straight to the dashboard and had every signed-out shot
        // silently capture Home instead. Clearing it here is what makes "anonymous" actually anonymous.
        await page.evaluateOnNewDocument(() => {
          try { localStorage.clear() } catch { /* first document may have no storage yet */ }
        })
      } else {
        await page.evaluateOnNewDocument((v1, auth) => {
          localStorage.setItem('reflexion.v1Session', JSON.stringify(v1))
          localStorage.setItem('reflexion.authSession', JSON.stringify(auth))
        }, V1_SESSION, AUTH_SESSION)
      }

      try {
        await page.goto(`${BASE}${screen.route}`, { waitUntil: 'networkidle2', timeout: 60_000 })
        if (screen.wait) {
          await page.waitForFunction((text) => document.body.innerText.includes(text), { timeout: 20_000 }, screen.wait)
            .catch(() => report.push(`  ? ${device.id}/${screen.id} never showed "${screen.wait}"`))
        }
        await sleep(1200) // let fonts settle and any focus effect refetch land
        // Expo's dev error toast is pinned to the bottom of the viewport and would cover the tab bar in
        // every shot. It reports react-native-web's nested-<button> warning, which is a web rendering
        // artifact of nested Pressables and not something a phone shows — noted in the report instead.
        await page.addStyleTag({ content: '#error-toast{display:none !important}' }).catch(() => {})
        const file = path.join(OUT, `${device.id}__${screen.id}.png`)
        await page.screenshot({ path: file })

        // Horizontal overflow is the failure mode that a screenshot hides, so measure it too.
        const overflow = await page.evaluate(() => ({
          scrollW: document.documentElement.scrollWidth,
          clientW: document.documentElement.clientWidth,
        }))
        if (overflow.scrollW > overflow.clientW + 1) {
          report.push(`  ✗ ${device.id}/${screen.id} OVERFLOWS: content ${overflow.scrollW}px in ${overflow.clientW}px`)
        }
      } catch (error) {
        report.push(`  ! ${device.id}/${screen.id} FAILED: ${String(error.message).slice(0, 120)}`)
      }
      await page.close()
      process.stdout.write(`  ${device.id}/${screen.id}\n`)
    }
    console.log(`done ${device.id}  (${device.w}x${device.h})  ${device.label}`)
  }

  await browser.close()
  const manifest = DEVICES.map((d) => `${d.id}  ${String(d.w).padStart(3)}x${String(d.h).padStart(4)}  dpr=${d.dpr}  ${d.label}`).join('\n')
  await writeFile(path.join(OUT, 'DEVICES.txt'), `${manifest}\n\nscreens:\n${SCREENS.map((s) => `  ${s.id}  ${s.route}`).join('\n')}\n`)
  console.log(`\n${DEVICES.length * SCREENS.length} shots -> ${OUT}`)
  if (report.length) {
    console.log('\nissues:')
    console.log([...new Set(report)].join('\n'))
    await writeFile(path.join(OUT, 'ISSUES.txt'), [...new Set(report)].join('\n'))
  } else {
    console.log('\nno overflow or page errors detected')
  }
}

main().catch((error) => { console.error(error); process.exit(1) })
