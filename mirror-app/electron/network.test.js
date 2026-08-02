// Pure-logic tests for the mirror's network service. Run with:
//   npm run test:network
//
// Only the pieces that do not touch nmcli/BlueZ are covered: the terse-output parser (where a bug hides
// whole Wi-Fi networks from the setup list) and the error translator (where a bug leaves an installer
// staring at raw nmcli stderr). The command wiring itself needs real hardware — see
// docs/mirror-app/network-setup.md for the on-device checklist.

const assert = require('node:assert/strict')
const test = require('node:test')

const { __testables } = require('./network')
const { splitTerse, friendlyError, tetherConnectionName, randomPassphrase } = __testables

test('splitTerse splits plain nmcli terse rows', () => {
  assert.deepEqual(splitTerse('*:HomeWifi:78:WPA2:2437'), ['*', 'HomeWifi', '78', 'WPA2', '2437'])
})

test('splitTerse keeps escaped colons inside an SSID', () => {
  // nmcli escapes a literal colon as \: — naive splitting turns one network into two broken rows.
  assert.deepEqual(splitTerse(':Bob\\:s iPhone:64:WPA2:5180'), ['', "Bob:s iPhone", '64', 'WPA2', '5180'])
})

test('splitTerse keeps escaped backslashes', () => {
  assert.deepEqual(splitTerse('a\\\\b:2'), ['a\\b', '2'])
})

test('splitTerse preserves empty leading and trailing fields', () => {
  // IN-USE is empty for every network the mirror is not currently on, and CONNECTION is empty for an
  // unmanaged device — dropping either shifts every later column.
  assert.deepEqual(splitTerse(':wifi:disconnected:'), ['', 'wifi', 'disconnected', ''])
})

test('friendlyError names a missing tool rather than blaming the network', () => {
  const message = friendlyError({ missing: true, stderr: 'spawn nmcli ENOENT' }, 'fallback')
  assert.match(message, /missing its network tools/)
})

test('friendlyError turns a rejected secret into password guidance', () => {
  const message = friendlyError({ missing: false, stderr: 'Error: Secrets were required, but not provided.' }, 'fallback')
  assert.match(message, /password was not accepted/)
})

test('friendlyError explains a polkit refusal as a permission problem', () => {
  const message = friendlyError({ missing: false, stderr: 'Error: Not authorized to control networking.' }, 'fallback')
  assert.match(message, /not allowed to change network settings/)
})

test('friendlyError falls back to the caller message for anything unrecognised', () => {
  assert.equal(friendlyError({ missing: false, stderr: 'something odd' }, 'Could not join HomeWifi.'), 'Could not join HomeWifi.')
})

test('tetherConnectionName is stable and safe for a NetworkManager profile name', () => {
  assert.equal(tetherConnectionName('AA:BB:CC:DD:EE:FF'), 'reflexion-bt-aabbccddeeff')
  assert.equal(tetherConnectionName('aa:bb:cc:dd:ee:ff'), tetherConnectionName('AA:BB:CC:DD:EE:FF'))
})

test('hotspot passphrase is WPA2-legal and avoids look-alike characters', () => {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    const passphrase = randomPassphrase()
    assert.ok(passphrase.length >= 8, 'WPA2 rejects passphrases shorter than 8 characters')
    // An installer reads this off the glass and types it into a phone, so 0/O and 1/l must not appear.
    assert.doesNotMatch(passphrase, /[0o1li]/)
  }
})
