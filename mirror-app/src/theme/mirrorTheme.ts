import { Platform } from 'react-native'

/**
 * Shared Mirror tokens for the final 9 August 2026 visual system.
 *
 * The reference screens are deliberately light: warm ivory, ink-blue type, sage/teal controls
 * and a restrained terracotta accent.  Keep these tokens central so the fourteen canonical surfaces
 * feel like one calm device rather than a collection of route-specific layouts.
 */
export const mirrorColors = {
  cream: '#FBF8F3',
  sand: '#F5F0E8',
  beige: '#E8E7DA',
  gold: '#E7D2B4',
  goldDeep: '#C87943',
  sage: '#B7C5AF',
  sageDeep: '#4F7067',
  taupe: '#C9B8A7',
  text: '#1E2B31',
  textSecondary: '#34464A',
  coral: '#C96D42',
  amber: '#B87943',
  white: '#FFFFFF',
  lineWarm: 'rgba(89, 83, 68, 0.18)',
  shadow: 'rgba(101, 75, 43, 0.14)',
  // Light translucent surfaces preserve the mirror's quiet, reflective feel without turning the UI dark.
  glassOverlay: 'rgba(255,255,255,0.62)',
  glassOverlayStrong: 'rgba(255,255,255,0.88)',

  // Compatibility aliases for the existing diagnostics and developer surfaces.
  ink: '#1E2B31',
  inkLift: '#F6F1E8',
  glass: 'rgba(255, 255, 255, 0.72)',
  glassStrong: 'rgba(255, 255, 255, 0.94)',
  line: 'rgba(118, 94, 62, 0.16)',
  linen: '#FFFFFF',
  linenSoft: '#617074',
  bronze: '#C87943',
  sageBright: '#4F7067',
  error: '#C96D42',
}

export const mirrorFonts = {
  display: Platform.select({ ios: 'New York', android: 'serif', default: 'serif' }),
  body: Platform.select({ ios: 'Avenir Next', android: 'sans-serif', default: 'sans-serif' }),
  bodyMedium: Platform.select({ ios: 'Avenir Next Medium', android: 'sans-serif-medium', default: 'sans-serif' }),
}
