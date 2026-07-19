import { Redirect } from 'expo-router';

export default function Index() {
  // In a real app, check auth state here and redirect accordingly.
  // For MVP, always go to main tabs.
  return <Redirect href="/(tabs)" />;
}
