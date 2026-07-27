import { Dimensions, PixelRatio } from 'react-native';

// The single source of truth for colour in the caregiver app.
//
// Before this module there were ~570 hard-coded hex literals across 19 separate StyleSheet.create blocks,
// so `#87566A` alone appeared 80 times. Two consequences worth naming, because they both actually
// happened: a palette change was a 19-file sweep, and when the WCAG audit found that the muted text colour
// only reached 2.70:1 on white, fixing it meant a scripted find-and-replace across a dozen screens rather
// than editing one line.
//
// Every value below is the value already shipping — this is an extraction, not a redesign.
//
// CONTRAST: the `text` tokens all clear WCAG AA 4.5:1 against every surface in `surface`. `textDecorative`
// clears 3:1 only and is for glyphs that duplicate an affordance (a chevron next to a tappable row), never
// for words a caregiver has to read. If you add a token, check it — scripts/checkContrast.mjs does it.

export const colors = {
  /**
   * Backgrounds text sits on. Every `text` token below is checked against every entry here, so a surface
   * that never carries text does not belong in this group — put it in `border` instead.
   */
  surface: {
    page: '#F8F3EC',
    card: '#FFFFFF',
    input: '#FBF8F4',
    muted: '#F4F0EA',
  },

  /** Text. All AA-compliant on every `surface` value. */
  text: {
    primary: '#2B2522',
    secondary: '#756C64',
    /** De-emphasised meta lines (dates, counts, sub-labels). */
    tertiary: '#766C61',
    onAccent: '#FFFFFF',
  },

  /** 3:1 only — decorative glyphs that repeat an affordance. Never body copy. */
  textDecorative: '#94897F',

  /** Brand accent: primary buttons, links, active tabs, spinners. */
  accent: '#87566A',
  accentPressed: '#6E2F48',

  border: {
    default: '#E7DED2',
    strong: '#D8CFC3',
    /** Hairline between rows inside a card. Only ever a border — never a text background. */
    subtle: '#F3EDE6',
  },

  /**
   * The inline "that didn't work" box on the auth screens. Previously hard-coded per screen, which is how
   * its border ended up at 1.53:1 against the card — invisible as a boundary. Reserved for a message the
   * caregiver must read and act on; it is never used for anything about their loved one's wellbeing, which
   * has its own non-alarming vocabulary in v1Status.ts.
   */
  error: {
    surface: '#FDECEC',
    border: '#AA7F7F',
    text: '#8A2E2E',
  },

  /** Unselected tab-bar label. Was #8D8278 (3.75:1 on white) — text, so it has to clear AA. */
  tabInactive: '#776C62',

  /** Form-field hint text. 3:1 by design: legible, but not mistakable for entered text. */
  placeholder: '#93887D',

  shadow: '#000000',
} as const;

/**
 * Status colour belongs to src/lib/v1Status.ts (STATUS_META), not here.
 *
 * It is the app's one authoritative status vocabulary — the muted "Option-1" palette fixed by the product
 * doc (§2.9) and keyed on the four server states including `establishing`. Re-declaring status colours in a
 * theme file is how a second, drifting vocabulary gets started, which is exactly the bug that let a patient
 * still learning their routine be painted red.
 */

// --- Responsive scaling -------------------------------------------------------------------------
// Android ships on a huge range of screen sizes; fixed-pixel type + spacing look cramped on a 360dp phone
// and lost on a 430dp+ phone or a tablet. Scale both to the device's SHORTER side against a 390dp design
// baseline, MODERATED (factor 0.5 — type moves half as much as raw width) and CLAMPED so a small phone
// never drops the type below the readability floor and a tablet never balloons. Computed once at module
// load: fine for phones, which don't resize; a split-screen tablet keeps the launch size until relaunch.
const BASELINE_WIDTH = 390;
const shortestSide = Math.min(Dimensions.get('window').width, Dimensions.get('window').height);
const widthScale = Math.min(1.3, Math.max(0.9, shortestSide / BASELINE_WIDTH));

/** Moderate, clamped, pixel-snapped scale of a base size. factor 0 = no scaling, 1 = full width scaling. */
export function scaleSize(size: number, factor = 0.5): number {
  return PixelRatio.roundToNearestPixel(size + (size * widthScale - size) * factor);
}
/** Type scale — same as scaleSize but never below the 12px readability floor (see fontSize note). */
function fontScale(size: number): number {
  return Math.max(12, scaleSize(size));
}

export const spacing = {
  xs: scaleSize(4),
  sm: scaleSize(8),
  md: scaleSize(12),
  lg: scaleSize(16),
  xl: scaleSize(20),
  xxl: scaleSize(28),
} as const;

export const radius = {
  sm: 8,
  md: 10,
  lg: 14,
  xl: 16,
  pill: 999,
} as const;

/**
 * 12 is the floor. Anything smaller is unreadable for the caregivers this app is for — often middle-aged
 * to older, checking the app one-handed — and clips first when the system font size is raised.
 */
export const fontSize = {
  caption: fontScale(12),
  body: fontScale(13),
  bodyLarge: fontScale(14),
  subheading: fontScale(15),
  heading: fontScale(17),
  title: fontScale(20),
  display: fontScale(26),
} as const;

/** Serif display face used for names, greetings and card titles. */
export const fontFamily = { display: 'Georgia' } as const;

/** The smallest reliably tappable target. Below this, use hitSlop rather than shrinking. */
export const MIN_TOUCH_TARGET = 44;

/** The soft elevation every card in the app shares. */
export const cardShadow = {
  shadowColor: colors.shadow,
  shadowOffset: { width: 0, height: 4 },
  shadowOpacity: 0.035,
  shadowRadius: 10,
  elevation: 2,
} as const;
