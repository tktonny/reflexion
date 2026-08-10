import { Redirect, useLocalSearchParams } from 'expo-router'

import { BootLoadingScreen, PairingScreen } from './index'
import { MirrorExperience, type MirrorVisualState } from '../src/components/mirror/MirrorExperience'

const states: MirrorVisualState[] = [
  'ambient', 'connecting', 'listening', 'heard', 'thinking', 'speaking', 'closing', 'saving',
  'offline', 'microphone_error', 'service_error',
]

/** Manual, development-only render harness for the portrait mirror acceptance matrix. */
export default function VisualAcceptanceScreen() {
  const { state: requestedState } = useLocalSearchParams<{ state?: string }>()
  if (!__DEV__) return <Redirect href="/" />
  if (requestedState === 'boot') {
    return <BootLoadingScreen checks={[]} />
  }
  if (requestedState === 'pairing') {
    return (
      <PairingScreen
        error=""
        onRetry={() => undefined}
        pairing={{
          deviceId: 'dev_mirror_acceptance',
          pairingId: 'pair_acceptance_482913',
          pairingCode: '482913',
          qrPayload: JSON.stringify({
            type: 'reflexion_device_pairing_v2',
            pairingId: 'pair_acceptance_482913',
            pairingCode: '482913',
          }),
        }}
      />
    )
  }
  const state = states.includes(requestedState as MirrorVisualState)
    ? requestedState as MirrorVisualState
    : 'ambient'
  const assistantText = state === 'closing'
    ? 'Thank you for chatting with me. I’ll let Chloe know you checked in today. Have a good day.'
    : 'That sounds lovely. What did you enjoy most about your morning?'
  return (
    <MirrorExperience
      assistantText={assistantText}
      bargeInActive={false}
      date="Wednesday, 22 July"
      greeting="Good morning"
      homeWidgets={[
        { icon: 'partly-sunny-outline', label: 'Partly cloudy', value: '24°C' },
        { icon: 'medkit-outline', label: 'Medication', value: '9:00 AM' },
      ]}
      microphoneActive={['listening', 'heard', 'speaking'].includes(state)}
      onEnd={() => undefined}
      onRetry={() => undefined}
      patientName="Margaret"
      progressText="TODAY’S CONVERSATION"
      state={state}
      statusText="A secure connection could not be established."
      time="8:15"
      userText={state === 'listening' ? '' : 'I went for a walk by the garden this morning.'}
      wakeError=""
      wakeListening
    />
  )
}
