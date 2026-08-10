/**
 * Canonical full-screen inventory from the final caregiver architecture.
 *
 * This is a product map, not a second router. Dynamic values are written as route
 * parameters so the same implementation can be reused from setup and Settings.
 * Overlays, sheets, native permission prompts and inline states intentionally do
 * not appear here.
 */
export type CanonicalScreen = { id: string; name: string; route: string };

export const CANONICAL_SCREENS: readonly CanonicalScreen[] = [
  { id: 'AUTH-01', name: 'Splash', route: '/splash' },
  { id: 'AUTH-02', name: 'Sign in', route: '/sign-in' },
  { id: 'AUTH-03', name: 'Forgot password', route: '/forgot-password' },
  { id: 'AUTH-04', name: 'Reset verification', route: '/reset-verification' },
  { id: 'AUTH-05', name: 'Reset password', route: '/reset-password' },
  { id: 'AUTH-06', name: 'Create account', route: '/create-account' },

  { id: 'SETUP-01', name: 'Welcome to Reflexion', route: '/welcome' },
  { id: 'SETUP-02', name: 'Setup overview', route: '/setup' },
  { id: 'SETUP-03', name: 'Loved-One Profile', route: '/setup/household' },
  { id: 'SETUP-04', name: 'Review household', route: '/setup/household-review' },
  { id: 'SETUP-05', name: 'Select loved one and device type', route: '/device/select' },
  { id: 'SETUP-06', name: 'Choose pairing method', route: '/device/[id]/pairing' },
  { id: 'SETUP-07', name: 'Scan QR code', route: '/device/[id]/qr' },
  { id: 'SETUP-08', name: 'Enter six-digit pairing code', route: '/device/[id]/code' },
  { id: 'SETUP-09', name: 'Placement guide', route: '/device/[id]/placement' },
  { id: 'SETUP-10', name: 'Device paired successfully', route: '/device/[id]/success' },
  { id: 'SETUP-11', name: 'Language & Accessibility', route: '/settings/language' },
  { id: 'SETUP-12', name: 'Voice preview', route: '/settings/voice-preview' },
  { id: 'SETUP-13', name: 'Routine category selection', route: '/settings/routines' },
  { id: 'SETUP-14', name: 'Routine examples and templates', route: '/settings/routines' },
  { id: 'SETUP-15', name: 'Add or edit routine', route: '/settings/routines' },
  { id: 'SETUP-16', name: 'Review routines', route: '/settings/routines' },
  { id: 'SETUP-17', name: 'Notification Preferences', route: '/settings/notifications' },
  { id: 'SETUP-18', name: 'Older-Adult Consent & Control overview', route: '/settings/consent' },
  { id: 'SETUP-19', name: 'Consent explanation', route: '/settings/consent' },
  { id: 'SETUP-20', name: 'Consent status', route: '/settings/consent' },
  { id: 'SETUP-21', name: 'Review setup', route: '/setup/review' },
  { id: 'SETUP-22', name: 'Setup complete', route: '/setup/complete' },

  { id: 'HOME-01', name: 'Home', route: '/(tabs)' },
  { id: 'LOVED-01', name: 'Loved-one dashboard — Today', route: '/loved-one/[id]' },
  { id: 'LOVED-02', name: 'Loved-one dashboard — Sessions', route: '/loved-one/[id]/sessions' },
  { id: 'LOVED-03', name: 'Session Detail', route: '/loved-one/[id]/sessions/[sessionId]' },
  { id: 'LOVED-04', name: 'Full Conversation', route: '/loved-one/[id]/sessions/[sessionId]/conversation' },
  { id: 'LOVED-05', name: 'Weekly Summary', route: '/loved-one/[id]/weekly-summary' },
  { id: 'LOVED-06', name: 'Trends', route: '/loved-one/[id]/trends' },
  { id: 'LOVED-07', name: 'History', route: '/loved-one/[id]/history' },
  { id: 'LOVED-08', name: 'Export Summaries', route: '/loved-one/[id]/export' },

  { id: 'ACT-01', name: 'Activity timeline', route: '/(tabs)/activity' },
  { id: 'ACT-02', name: 'Activity event detail', route: '/activity/[eventId]' },

  { id: 'CHAT-01', name: 'Chat list', route: '/(tabs)/chat' },
  { id: 'CHAT-02', name: 'Chat thread', route: '/chat/[id]' },
  { id: 'CHAT-03', name: 'Text message composer', route: '/chat/[id]/compose' },
  { id: 'CHAT-04', name: 'Photo message composer', route: '/chat/[id]/photo' },
  { id: 'CHAT-05', name: 'Voice message recorder', route: '/chat/[id]/voice' },
  { id: 'CHAT-06', name: 'Message preview', route: '/chat/[id]/preview' },

  { id: 'SET-01', name: 'Settings overview', route: '/(tabs)/settings' },
  { id: 'SET-02', name: 'Account', route: '/settings/account' },
  { id: 'SET-03', name: 'App Language', route: '/settings/app-language' },
  { id: 'SET-04', name: 'Loved Ones', route: '/settings/household' },
  { id: 'SET-05', name: 'Edit Loved-One Profile', route: '/settings/household/[id]' },
  { id: 'SET-06', name: 'Away Mode', route: '/settings/away' },
  { id: 'SET-07', name: 'Routine Management', route: '/settings/routines' },
  { id: 'SET-08', name: 'Connected Devices', route: '/settings/devices' },
  { id: 'SET-09', name: 'Device Detail and Status', route: '/device/[id]/detail' },
  { id: 'SET-10', name: 'Device Troubleshooting', route: '/device/[id]/troubleshooting' },
  { id: 'SET-11', name: 'Privacy & Data', route: '/settings/privacy' },
  { id: 'SET-12', name: 'View Product Consent', route: '/settings/consent' },
  { id: 'SET-13', name: 'Product Consent History', route: '/settings/privacy' },
  { id: 'SET-14', name: 'Data Retention', route: '/settings/privacy' },
  { id: 'SET-15', name: 'Vendor and Cross-Border Disclosure', route: '/settings/privacy' },
  { id: 'SET-16', name: 'FAQ & Help', route: '/settings/help' },
  { id: 'SET-17', name: 'Contact Support', route: '/settings/contact-support' },
  { id: 'SET-18', name: 'Feedback', route: '/settings/feedback' },
  { id: 'SET-19', name: 'Subscription', route: '/settings/subscription' },
  { id: 'SET-20', name: 'Payment Method', route: '/settings/payment-method' },
  { id: 'SET-21', name: 'About Reflexion', route: '/settings/about' },

  { id: 'RES-01', name: 'Research Participation overview', route: '/research/overview' },
  { id: 'RES-02', name: 'Study information and consent', route: '/research/study' },
  { id: 'RES-03', name: 'Research consent confirmation', route: '/research/confirmation' },
  { id: 'RES-04', name: 'Research Participation status', route: '/settings/research' },
] as const;

export const CANONICAL_SCREEN_COUNT = CANONICAL_SCREENS.length;

