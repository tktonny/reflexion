export type FAQItem = {
  id: string;
  question: string;
  answer: string;
};

export type FAQGroup = {
  category: string;
  items: FAQItem[];
};

/** Help Centre copy for the caregiver app. Answers describe the current product and contracts. */
export const FAQ_GROUPS: FAQGroup[] = [
  {
    category: 'Getting started',
    items: [
      {
        id: 'getting-started-purpose',
        question: 'What can I use the caregiver app for?',
        answer: 'Use the app to manage loved-one profiles, connect a Reflexion Mirror, set up routines and notifications, review recorded activity and sessions, and send family messages.',
      },
      {
        id: 'getting-started-setup',
        question: 'Where do I continue an unfinished setup?',
        answer: 'Sign in and open the setup steps shown for your account. You can also review the same settings later from Settings, including loved ones, devices, routines, notifications, consent and research participation.',
      },
    ],
  },
  {
    category: 'Pairing the Mirror',
    items: [
      {
        id: 'pairing-start',
        question: 'How do I pair a Mirror?',
        answer: 'Open Settings, choose Connected Devices and select Pair a Mirror. Follow the pairing steps using the current code shown on the physical Mirror, then choose the loved one it belongs to.',
      },
      {
        id: 'pairing-fails',
        question: 'What if pairing does not work?',
        answer: 'Make sure the Mirror is powered on and that you are using its current pairing code. Retry the pairing step after checking the device connection. Contact support if the code has expired or the problem continues.',
      },
    ],
  },
  {
    category: 'Routines and reminders',
    items: [
      {
        id: 'routines-how',
        question: 'How do routines and reminders work?',
        answer: 'Create gentle daily prompts for a loved one in Settings > Routines. The Mirror reports responses; the app does not treat a missing response as proof that a routine happened.',
      },
      {
        id: 'routines-notifications',
        question: 'How do I change routine notifications?',
        answer: 'Open Settings > Routines, choose the loved one and edit a routine. You can choose compatible notification options for a missed or unclear response and for the daily summary.',
      },
    ],
  },
  {
    category: 'Family messages',
    items: [
      {
        id: 'messages-send',
        question: 'How do I send a family message?',
        answer: 'Open the Chat tab, choose a loved one and select the message action you want to use. A text message is sent to the paired Mirror, and the app shows the delivery or viewed status returned by the product.',
      },
      {
        id: 'messages-status',
        question: 'What does a message status mean?',
        answer: 'The status describes the message record, such as queued, delivered or viewed. Viewed means your loved one chose to view the message on the Mirror; it is not inferred from delivery alone.',
      },
    ],
  },
  {
    category: 'Notifications',
    items: [
      {
        id: 'notifications-manage',
        question: 'Where do I manage app notifications?',
        answer: 'Open Settings > App notifications. When Android asks for notification permission, choose Allow if you want caregiver updates. If permission was denied, use Open phone settings to turn it back on.',
      },
      {
        id: 'notifications-missing',
        question: 'Why did I not receive a notification?',
        answer: 'Check that notifications are enabled in the app and in your phone settings. Also check the notification choices for the relevant routine or update. If notifications still do not arrive, contact support.',
      },
    ],
  },
  {
    category: 'Consent & privacy',
    items: [
      {
        id: 'consent-choice',
        question: 'Who makes the consent choice?',
        answer: 'The choice belongs to the loved one. A caregiver can help explain the information and record the loved one’s choice in the app or on the Mirror. Product consent and optional research participation are separate.',
      },
      {
        id: 'consent-review',
        question: 'Where can I review consent and privacy choices?',
        answer: 'Open Settings > Consent & control to review product consent, or Settings > Privacy & data to review consent history, retention information and available data requests.',
      },
    ],
  },
  {
    category: 'Research participation',
    items: [
      {
        id: 'research-optional',
        question: 'What is research participation?',
        answer: 'Reflexion is working with NUS Yong Loo Lin School of Medicine and NUHS on research into healthy ageing and cognitive health. Participation is optional, separate from ordinary Reflexion use and can be declined without affecting the product.',
      },
      {
        id: 'research-information',
        question: 'Where can I read the study information?',
        answer: 'Open Settings > Research participation and choose View study information. The page shows the current collaboration information available in Reflexion without asking you to decide before you are ready.',
      },
    ],
  },
  {
    category: 'Device troubleshooting',
    items: [
      {
        id: 'device-offline',
        question: 'What should I do if the Mirror may be offline?',
        answer: 'Open Settings > Connected Devices and select the relevant Mirror to review its technical status. Check that it has power and that the Mirror reports an internet connection; contact support if the status remains unavailable.',
      },
      {
        id: 'device-audio',
        question: 'How can I check the Mirror audio?',
        answer: 'Open the device troubleshooting steps to review the connection, microphone and speaker status reported by the Mirror. Keep the device powered and unobstructed, then contact support if a check remains unavailable.',
      },
    ],
  },
  {
    category: 'Account & sign-in',
    items: [
      {
        id: 'account-details',
        question: 'Where can I update my account details?',
        answer: 'Open Settings > Account to review your personal details and sign-in methods. Verification may be required before a new email address or phone number is saved.',
      },
      {
        id: 'account-forgot-password',
        question: 'What should I do if I forgot my password?',
        answer: 'Choose Forgot password on the sign-in screen and follow the verification steps. Contact support if you cannot access the recovery method for your account.',
      },
    ],
  },
];
