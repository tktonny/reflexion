import { Redirect } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, View } from 'react-native';
import { loadV1Session } from '../src/lib/v1AuthSession';
import { loadPendingVerification } from '../src/lib/pendingVerification';
import { colors } from '../src/theme';

export default function Index() {
  const [ready, setReady] = useState(false); const [signedIn, setSignedIn] = useState(false); const [pendingVerification, setPendingVerification] = useState(false);
  useEffect(() => { void Promise.all([loadV1Session(), loadPendingVerification()]).then(([session, pending]) => { setSignedIn(Boolean(session)); setPendingVerification(!session && Boolean(pending)); setReady(true); }); }, []);
  if (!ready) return <View style={{ alignItems: 'center', backgroundColor: colors.surface.page, flex: 1, justifyContent: 'center' }}><ActivityIndicator color={colors.accent} /></View>;
  return <Redirect href={signedIn ? '/(tabs)' : pendingVerification ? '/account-verification' : '/sign-in'} />;
}
