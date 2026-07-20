Pod::Spec.new do |s|
  s.name           = 'ExpoPcmAudio'
  s.version        = '0.1.0'
  s.summary        = 'Streaming PCM capture/playback for Qwen realtime audio'
  s.description    = 'Native PCM16 mic capture (16 kHz) and streaming playback (24 kHz) for the v3 direct realtime WS.'
  s.author         = ''
  s.homepage       = 'https://docs.expo.dev/modules/'
  s.platforms      = { :ios => '15.1' }
  s.source         = { git: '' }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'SWIFT_COMPILATION_MODE' => 'wholemodule'
  }

  s.source_files = "**/*.{h,m,mm,swift,hpp,cpp}"
end
