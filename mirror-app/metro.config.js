// Expo default Metro config, extended so the openWakeWord ONNX models under assets/wakeword/ are
// bundled as assets (Metro doesn't treat .onnx as an asset by default).
const { getDefaultConfig } = require('expo/metro-config')

const config = getDefaultConfig(__dirname)
if (!config.resolver.assetExts.includes('onnx')) {
  config.resolver.assetExts.push('onnx')
}

module.exports = config
