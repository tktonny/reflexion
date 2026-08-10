// Context-isolated preload for the Linux appliance. The renderer receives only named, validated network
// operations; it never receives a generic IPC invoke surface or the raw IpcRendererEvent.
const { contextBridge, ipcRenderer } = require('electron')

const invoke = (channel) => (options) => ipcRenderer.invoke(channel, options ?? {})

contextBridge.exposeInMainWorld('reflexionMirror', {
  platform: 'linux-electron',
  // The Electron shell proxies /api from its loopback origin to the configured backend.
  apiProxy: true,
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
  setupState: invoke('reflexion:setup:state'),
  setupStart: invoke('reflexion:setup:start'),
  setupStop: invoke('reflexion:setup:stop'),
  onSetupState: (listener) => {
    if (typeof listener !== 'function') return () => {}
    const handler = (_event, payload) => listener(payload)
    ipcRenderer.on('reflexion:setup:state', handler)
    return () => ipcRenderer.removeListener('reflexion:setup:state', handler)
  },
})
