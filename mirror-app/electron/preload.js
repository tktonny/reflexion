// Minimal, context-isolated preload. The web app reaches the relay over a WebSocket directly, so no
// privileged main-process bridge is needed today. This is kept as the seam for future Linux-native
// integration (e.g. a desktop wake-word module or GPIO), which would be exposed here rather than by
// disabling context isolation.
const { contextBridge } = require('electron')

contextBridge.exposeInMainWorld('reflexionMirror', {
  platform: 'linux-electron',
})
