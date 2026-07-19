# Reflexion Mobile Mirror Interface

Expo React Native port of the Reflexion v2 mirror verification and realtime conversation flow.

## What Was Ported

- Mirror verification updates `NursePatientConfig.patients.$.mirrorVerified` in MongoDB.
- Verified `NursePatientConfig`, active mirror ID, nurse ID, and patient ID are cached on-device with AsyncStorage.
- Realtime conversation uses OpenAI Realtime WebRTC, the same prompt flow, server VAD, Whisper transcription, transcript metrics, latency metrics, and conversation save payload as `reflexion-v2`.
- Expo Router API routes in `app/api` handle MongoDB and OpenAI calls inside this app, matching `reflexion-native-app`.

## Environment

App root `.env`:

- `MONGODB_URI`: same MongoDB connection string used by `reflexion-v2`.
- `OPENAI_API_KEY`: OpenAI API key used to mint realtime client secrets.

Do not prefix these with `EXPO_PUBLIC_`. They are read by Expo Router API routes.

## Run

```sh
npm install
npm run dev
```

`react-native-webrtc` requires a development build for iOS/Android; it will not run in plain Expo Go.
