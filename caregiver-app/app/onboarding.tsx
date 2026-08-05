import { Redirect } from 'expo-router';
/** The obsolete four-step setup has been replaced by create-account → welcome → independent setup categories. */
export default function OnboardingRedirect() { return <Redirect href="/create-account" />; }
