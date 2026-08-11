// Context-isolated preload for the Linux (Ubuntu) mirror.
//
// Two things cross this boundary, and nothing else:
//
//   1. `reflexionMirror` — a small identity/config object. `apiProxy: true` tells the renderer that the
//      Electron shell forwards /api and /health from its own loopback origin, so the app must use
//      SAME-ORIGIN relative URLs instead of an absolute backend origin (see src/config/apiUrl.ts). This
//      is what keeps the mirror out of the CORS preflight that silently blocked a shipped AppImage.
//
//   2. `reflexionNetwork` — the privileged network setup surface (Wi-Fi, hotspot, Bluetooth tethering).
//      A sandboxed Chromium page cannot configure an appliance's network, and an Ubuntu unit arrives in a
//      home with no network at all, so the app needs a way to ask the main process to do it.
//
// Each method is a fixed, named IPC channel — deliberately NOT a generic "invoke(channel, args)" bridge,
// which would hand the renderer the whole main-process IPC surface.
const { contextBridge, ipcRenderer } = require('electron')

const invoke = (channel) => (options) => ipcRenderer.invoke(channel, options ?? {})

contextBridge.exposeInMainWorld('reflexionMirror', {
  platform: 'linux-electron',
  // The shell proxies backend calls; the renderer must therefore NOT prefix an absolute API origin.
  apiProxy: true,
})

// The unit's identity, read at runtime from <userData>/device-config.json rather than compiled in — one
// AppImage has to serve a whole fleet, and a bootstrap token is device-bound (see electron/deviceConfig.js).
contextBridge.exposeInMainWorld('reflexionProvisioning', {
  bootstrapToken: invoke('reflexion:provisioning:bootstrap-token'),
})

// OTA. expo-updates has no web implementation, so the Linux build cannot use the Android update path at all;
// the shell does the work and the renderer drives it from the same settings screen. Two channels: the
// renderer bundle (small, frequent) and the Electron shell itself (large, rare).
contextBridge.exposeInMainWorld('reflexionUpdates', {
  state: invoke('reflexion:updates:state'),
  bundleCheck: invoke('reflexion:updates:bundle-check'),
  bundleApply: invoke('reflexion:updates:bundle-apply'),
  bundleRollback: invoke('reflexion:updates:bundle-rollback'),
  shellCheck: invoke('reflexion:updates:shell-check'),
  shellApply: invoke('reflexion:updates:shell-apply'),
  shellDiscard: invoke('reflexion:updates:shell-discard'),
  reload: invoke('reflexion:updates:reload'),
  relaunch: invoke('reflexion:updates:relaunch'),
  // Reports that this bundle actually rendered. The shell rolls back on the next launch without it, so it
  // must be called unconditionally on boot — see src/lib/otaUpdates.ts.
  markBooted: invoke('reflexion:updates:booted'),
})

contextBridge.exposeInMainWorld('reflexionNetwork', {
  capabilities: invoke('reflexion:network:capabilities'),
  status: invoke('reflexion:network:status'),
  wifiScan: invoke('reflexion:wifi:scan'),
  wifiConnect: invoke('reflexion:wifi:connect'),
  wifiConnectSaved: invoke('reflexion:wifi:connect-saved'),
  wifiForget: invoke('reflexion:wifi:forget'),
  wifiSetRadio: invoke('reflexion:wifi:radio'),
  hotspotStart: invoke('reflexion:hotspot:start'),
  hotspotStop: invoke('reflexion:hotspot:stop'),
  bluetoothStatus: invoke('reflexion:bluetooth:status'),
  bluetoothSetPower: invoke('reflexion:bluetooth:power'),
  bluetoothScan: invoke('reflexion:bluetooth:scan'),
  bluetoothPair: invoke('reflexion:bluetooth:pair'),
  bluetoothTether: invoke('reflexion:bluetooth:tether'),
  bluetoothDisconnect: invoke('reflexion:bluetooth:disconnect'),

  // Setup mode. The mirror has no keyboard and no mouse, so when it cannot reach the network it broadcasts
  // its own hotspot and serves a phone-facing setup page; the mirror's SCREEN has to display the hotspot
  // name, passphrase, portal address and PIN, which is what the renderer reads through here.
  setupState: invoke('reflexion:setup:state'),
  setupStart: invoke('reflexion:setup:start'),
  setupStop: invoke('reflexion:setup:stop'),
  /**
   * Subscribe to setup-state changes. Push rather than poll because the important transition — the join
   * result — happens exactly when the phone's connection has died, so the screen is the only thing that can
   * report it, and it must do so immediately. Returns an unsubscribe function.
   */
  onSetupState: (listener) => {
    if (typeof listener !== 'function') return () => {}
    // Only the payload crosses the bridge — never the raw IpcRendererEvent, which carries `sender`.
    const handler = (_event, payload) => listener(payload)
    ipcRenderer.on('reflexion:setup:state', handler)
    return () => ipcRenderer.removeListener('reflexion:setup:state', handler)
  },
})
