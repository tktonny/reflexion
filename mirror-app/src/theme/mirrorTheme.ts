import { Platform } from 'react-native'

/**
 * Mirror palette from the June–August 2026 consumer design specification.
 * The legacy aliases are kept because developer-only hardware screens still import them.
 */
export const mirrorColors = {
  cream: '#FFF9F1',
  sand: '#F6F1E8',
  beige: '#EDE5D6',
  gold: '#E7CFA6',
  goldDeep: '#B98954',
  sage: '#ABC5A1',
  sageDeep: '#637B5F',
  taupe: '#BBAFA0',
  text: '#282828',
  textSecondary: '#686868',
  coral: '#C9786E',
  amber: '#C9954D',
  white: '#FFFFFF',
  lineWarm: 'rgba(118, 94, 62, 0.16)',
  shadow: 'rgba(101, 75, 43, 0.16)',

  // Compatibility aliases for the existing diagnostics and developer surfaces.
  ink: '#282828',
  inkLift: '#F6F1E8',
  glass: 'rgba(255, 255, 255, 0.68)',
  glassStrong: 'rgba(255, 255, 255, 0.88)',
  line: 'rgba(118, 94, 62, 0.16)',
  linen: '#282828',
  linenSoft: '#686868',
  bronze: '#B98954',
  sageBright: '#637B5F',
  error: '#C9786E',
}

export const mirrorFonts = {
  display: Platform.select({ ios: 'New York', android: 'serif', default: 'serif' }),
  body: Platform.select({ ios: 'Avenir Next', android: 'sans-serif', default: 'sans-serif' }),
  bodyMedium: Platform.select({ ios: 'Avenir Next Medium', android: 'sans-serif-medium', default: 'sans-serif' }),
}
