import { Redirect, useLocalSearchParams } from 'expo-router';
import React from 'react';

/** Delivery status is an inline state in the chat thread, not a canonical screen. */
export default function MessageStatusRedirect() {
  const { id } = useLocalSearchParams<{ id: string }>();
  return <Redirect href={id ? `/chat/${id}` : '/(tabs)/chat'} />;
}
