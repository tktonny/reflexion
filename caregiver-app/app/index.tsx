import { Redirect } from 'expo-router';
import { getStoredAuthSession } from '../src/lib/authSession';

// Auth-aware entry redirect. The session is already hydrated before this renders — AuthGate
// (app/_layout.tsx) withholds <Stack> until loadStoredAuthSession() has run — so this synchronous
// read is safe and decides the first screen without a flash of Home for signed-out users.
export default function Index() {
  const session = getStoredAuthSession();
  return <Redirect href={session ? '/(tabs)' : '/sign-in'} />;
}
