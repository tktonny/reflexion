import { useContext } from 'react';
// Deep import: expo-router 56 vendors react-navigation's bottom-tabs rather than depending on it, so
// `@react-navigation/bottom-tabs` does not resolve in this project. The context is the public-ish surface of
// that vendored copy; if an expo-router upgrade moves it, this falls back to the theme constant rather than
// crashing, and the fallback is only wrong by the safe-area inset.
import { BottomTabBarHeightContext } from 'expo-router/build/react-navigation/bottom-tabs';
import { tabBarClearanceFallback } from '../theme';

/**
 * How much room a scrolling tab screen must leave at the bottom of its content.
 *
 * Prefer the navigator's own measured height: it already includes the safe-area inset and whatever the bar
 * grew to at the current system font size. Only when this hook is used outside a tab navigator — a pushed
 * screen, or a test — does it fall back to the theme's computed clearance.
 *
 * The measured height is the bar itself, so a little breathing room is added on top; content that stops
 * exactly at the bar's top edge reads as clipped even when it is technically complete.
 */
export function useTabBarClearance(): number {
  const measured = useContext(BottomTabBarHeightContext);
  // The measured height is the bar itself; a proportional margin on top keeps content that ends exactly at
  // its edge from reading as clipped, and scales with the bar instead of being a typed constant.
  return typeof measured === 'number' ? Math.round(measured * 1.12) : tabBarClearanceFallback();
}
