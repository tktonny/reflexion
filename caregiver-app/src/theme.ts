import { Dimensions, PixelRatio, Platform } from 'react-native';

/**
 * Reflexion Caregiver App — Design System
 *
 * Based on the 2026-08-02 architecture spec and design direction.
 * Warm white background, deep teal accent, soft states that never feel clinical.
 */

export const colors = {
  surface: {
    page: '#FBF8F2',
    card: '#FFFDFC',
    input: '#FFFDFC',
    muted: '#F6F1E8',
  },

  text: {
    primary: '#16324A',
    secondary: '#606D77',
    tertiary: '#626E75',
    onAccent: '#FFFFFF',
  },

  /** Brand accent — Reflexion teal. Buttons, links, active tabs. */
  accent: '#0C746D',
  accentPressed: '#075D58',

  /** Low-saturation accents used by factual alerts; never imply a health judgement. */
  alertAccent: {
    laterThanUsual: '#A66A22',
    connection: '#718078',
  },

  /** Status colors — named for what they mean, per architecture §2.4 */
  status: {
    /** Interaction recorded today */
    green: '#347A3B',
    greenBg: '#EAF4E7',
    /** No interaction yet / shorter than usual */
    amber: '#A86200',
    amberBg: '#FFF2DD',
    /** Device may be offline / technical uncertainty */
    grey: '#68727B',
    greyBg: '#F0F2F3',
    /** Needs your attention */
    red: '#B42318',
    redBg: '#FDEBE8',
  },

  border: {
    default: '#E9E1D6',
    strong: '#D9D3CA',
    subtle: '#ECE5DC',
  },

  error: {
    surface: '#FEE2E2',
    border: '#AD5555',
    text: '#991B1B',
  },

  tabInactive: '#596568',

  /** Decorative icons and chevrons; deliberately quieter than readable secondary text. */
  textDecorative: '#728C6F',

  placeholder: '#737E86',

  shadow: '#000000',
} as const;

// --- Responsive scaling ---
const BASELINE_WIDTH = 390;
const shortestSide = Math.min(Dimensions.get('window').width, Dimensions.get('window').height);
const widthScale = Math.min(1.3, Math.max(0.9, shortestSide / BASELINE_WIDTH));

export function scaleSize(size: number, factor = 0.5): number {
  return PixelRatio.roundToNearestPixel(size + (size * widthScale - size) * factor);
}

function fontScale(size: number): number {
  return Math.max(12, scaleSize(size));
}

export const spacing = {
  xs: scaleSize(4),
  sm: scaleSize(8),
  md: scaleSize(12),
  lg: scaleSize(16),
  xl: scaleSize(20),
  xxl: scaleSize(32),
  editorial: scaleSize(40),
  welcome: scaleSize(48),
  /** The single horizontal boundary used by every screen and sheet. */
  screen: scaleSize(24, 0.35),
} as const;

export const radius = {
  sm: 8,
  md: 12,
  lg: 14,
  xl: 18,
  pill: 999,
} as const;

export const fontSize = {
  caption: fontScale(12),
  body: fontScale(14),
  bodyLarge: fontScale(16),
  subheading: fontScale(17),
  heading: fontScale(20),
  title: fontScale(28),
  display: fontScale(34),
} as const;

/** System font stack — clean, readable, available everywhere. */
export const fontFamily = {
  /** Editorial display face for warm, human headings; body copy remains the platform font. */
  display: Platform.select({ android: 'serif', ios: 'Georgia', default: 'Georgia' }) || 'serif',
  regular: 'System',
  medium: 'System',
  semibold: 'System',
  bold: 'System',
} as const;

export const fontWeight = {
  regular: '400' as const,
  medium: '500' as const,
  semibold: '600' as const,
  bold: '700' as const,
};

export const MIN_TOUCH_TARGET = 44;

export const cardShadow = {
  shadowColor: '#16324A',
  shadowOffset: { width: 0, height: 5 },
  shadowOpacity: 0.055,
  shadowRadius: 20,
  elevation: 2,
} as const;

export function tabIconSize(): number {
  return Math.round(fontSize.body * 1.5 * PixelRatio.getFontScale());
}

export function tabBarContentHeight(): number {
  const labelHeight = fontSize.caption * 1.45 * PixelRatio.getFontScale();
  const navigatorInternalGap = spacing.xs;
  return Math.round(tabIconSize() + labelHeight + spacing.sm + spacing.xs + navigatorInternalGap);
}

export function tabBarClearanceFallback(): number {
  return Math.round(tabBarContentHeight() * 1.12);
}

export const maxContentWidth = Math.round(fontSize.body * 0.6 * 65 + spacing.xl * 2);

export const contentColumn = {
  alignSelf: 'center' as const,
  maxWidth: maxContentWidth,
  width: '100%' as const,
};

/** Shared page geometry for safe-area, keyboard-aware screens. */
export const layout = {
  horizontalPadding: spacing.screen,
  verticalPadding: spacing.lg,
  bottomPadding: spacing.welcome,
  keyboardOffset: 0,
} as const;
