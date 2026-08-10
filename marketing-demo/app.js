const ASSET = (name) => "/assets/" + name;

const routes = {
  "/": { label: "Home", className: "page--home" },
  "/how-it-works": { label: "How it works", className: "page--process" },
  "/for-families": { label: "For families", className: "page--families" },
  "/the-mirror": { label: "The mirror", className: "page--mirror" },
  "/about": { label: "About", className: "page--about" },
};

const navItems = [
  { label: "How it works", path: "/how-it-works" },
  { label: "For families", path: "/for-families" },
  { label: "The mirror", path: "/the-mirror" },
  { label: "About", path: "/about" },
];

const mirrorScreens = {
  idle: {
    source: "mirror-idle.png",
    alt: "Reflexion Mirror idle home screen with a daily check-in",
  },
  speaking: {
    source: "mirror-speaking.png",
    alt: "Reflexion Mirror speaking screen",
  },
  routine: {
    source: "mirror-routine.png",
    alt: "Reflexion Mirror routine reminder screen",
  },
  message: {
    source: "mirror-message.png",
    alt: "Reflexion Mirror family message screen",
  },
};

const phoneScreens = {
  home: {
    source: "caregiver-home.png",
    alt: "Reflexion caregiver app home screen",
  },
  trends: {
    source: "caregiver-trends.png",
    alt: "Reflexion caregiver app trends screen",
  },
  chat: {
    source: "caregiver-chat.png",
    alt: "Reflexion caregiver app chat screen",
  },
  routines: {
    source: "caregiver-routines.png",
    alt: "Reflexion caregiver app routine setup screen",
  },
};

const homeAssets = {
  logo: "ASSET-logo.jpg",
  mirror: "ASSET-reflexion-mirror.png",
  caregiver: "ASSET-caregiver-home.png",
};

function currentPath() {
  const path = window.location.pathname.replace(/\/+$/, "") || "/";
  return routes[path] ? path : "/";
}

function icon(name, className) {
  const paths = {
    arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
    arrowUp: '<path d="M12 19V5m-6 6 6-6 6 6"/>',
    bell: '<path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9ZM10 21h4"/>',
    brain: '<path d="M9.5 4.5A3.5 3.5 0 0 0 6 8v.5A3.5 3.5 0 0 0 4.5 15 3.5 3.5 0 0 0 8 18.5h1.5V20M14.5 4.5A3.5 3.5 0 0 1 18 8v.5a3.5 3.5 0 0 1 1.5 6.5 3.5 3.5 0 0 1-3.5 3.5h-1.5V20M9.5 8h5M9 12h6M9.5 16h5"/>',
    calendar: '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 10h18"/>',
    chart: '<path d="M4 19V5M4 19h17"/><path d="m7 15 3-3 3 2 5-6"/>',
    check: '<path d="m5 12 4 4L19 6"/>',
    close: '<path d="m6 6 12 12M18 6 6 18"/>',
    dots: '<circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/>',
    document: '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v5h5M9 12h6M9 16h6"/>',
    heart: '<path d="M20.8 8.7c0 5.4-8.8 10.1-8.8 10.1S3.2 14.1 3.2 8.7A4.6 4.6 0 0 1 12 6.3a4.6 4.6 0 0 1 8.8 2.4Z"/>',
    home: '<path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V10Z"/><path d="M9 21v-7h6v7"/>',
    leaf: '<path d="M20 4C10 4 4 8.2 4 15.5 4 18.8 6.2 21 9.5 21 16.8 21 20 13.8 20 4Z"/><path d="M4 20c3.5-4.2 7.4-6.8 12-8.2"/>',
    link: '<path d="m9 15-1.5 1.5a4 4 0 0 1-5.7-5.7L5 7.6a4 4 0 0 1 5.7 0M15 9l1.5-1.5a4 4 0 0 1 5.7 5.7L19 16.4a4 4 0 0 1-5.7 0M8 12h8"/>',
    message: '<path d="M20 11.5a7.5 7.5 0 0 1-8 7.5 8.4 8.4 0 0 1-3.3-.7L4 20l1.7-3.9A7.2 7.2 0 0 1 4 11.5 7.5 7.5 0 0 1 12 4a7.5 7.5 0 0 1 8 7.5Z"/><path d="M8 12h.01M12 12h.01M16 12h.01"/>',
    mic: '<rect x="8" y="3" width="8" height="12" rx="4"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3M8 21h8"/>',
    people: '<circle cx="9" cy="8" r="3"/><circle cx="17" cy="10" r="2.5"/><path d="M3 19a6 6 0 0 1 12 0M15 16a5 5 0 0 1 6 3"/>',
    play: '<path d="m8 5 11 7-11 7V5Z"/>',
    shield: '<path d="M12 3 20 6v5c0 5-3.4 8.5-8 10-4.6-1.5-8-5-8-10V6l8-3Z"/><path d="m8.5 12 2.3 2.3 4.8-5"/>',
    spark: '<path d="m12 2 1.7 6.3L20 10l-6.3 1.7L12 18l-1.7-6.3L4 10l6.3-1.7L12 2ZM19 16l.6 2.4L22 19l-2.4.6L19 22l-.6-2.4L16 19l2.4-.6L19 16Z"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    wave: '<path d="M3 12h2M7 8v8M11 5v14M15 8v8M19 10v4M22 12h-1"/>',
  };
  return (
    '<svg class="icon ' +
    (className || "") +
    '" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    (paths[name] || paths.spark) +
    "</svg>"
  );
}

function badge(label) {
  return '<span class="eyebrow">' + icon("leaf") + "<span>" + label + "</span></span>";
}

function button(label, kind, className) {
  return (
    '<button type="button" class="button ' +
    (kind ? "button--" + kind + " " : "") +
    (className || "") +
    '" data-open-waitlist>' +
    label +
    "</button>"
  );
}

function formButton(label) {
  return '<button type="submit" class="button button--small">' + label + "</button>";
}

function routeLink(label, path, className) {
  return (
    '<a class="' +
    (className || "") +
    '" href="' +
    path +
    '" data-route="' +
    path +
    '">' +
    label +
    "</a>"
  );
}

function brand() {
  return (
    '<a class="brand" href="/" data-route="/" aria-label="Reflexion home">' +
    '<span class="brand-mark">' +
    icon("leaf") +
    "</span>" +
    '<span class="brand-name">Reflexion</span>' +
    "</a>"
  );
}

function homeBrand() {
  return (
    '<a class="brand home-brand" href="/" data-route="/" aria-label="Reflexion home">' +
    '<img src="' +
    ASSET(homeAssets.logo) +
    '" alt="Reflexion" />' +
    "</a>"
  );
}

function header() {
  const path = currentPath();
  const isHome = path === "/";
  const links = navItems
    .map((item) => {
      return (
        '<a href="' +
        item.path +
        '" data-route="' +
        item.path +
        '" class="' +
        (path === item.path ? "is-active" : "") +
        '">' +
        item.label +
        "</a>"
      );
    })
    .join("");

  return (
    '<header class="site-header' +
    (isHome ? " home-header" : "") +
    '">' +
    (isHome ? homeBrand() : brand()) +
    '<nav class="site-nav" aria-label="Primary">' +
    links +
    "</nav>" +
    button("Join the waitlist", "", "button--small") +
    '<button class="nav-toggle" type="button" aria-label="Open navigation" aria-expanded="false">' +
    icon("dots") +
    "</button>" +
    "</header>"
  );
}

function footer() {
  return (
    '<footer class="footer">' +
    "<div>" +
    brand() +
    '<p class="footer-intro">A gentle daily rhythm for loved ones and the families who care about them.</p>' +
    '<span class="footer-small">Local editable demo · No waitlist data is sent</span>' +
    "</div>" +
    '<div><p class="footer-heading">Explore</p><div class="footer-links">' +
    routeLink("How it works", "/how-it-works") +
    routeLink("For families", "/for-families") +
    routeLink("The mirror", "/the-mirror") +
    routeLink("About", "/about") +
    "</div></div>" +
    '<div><p class="footer-heading">Start here</p><div class="footer-links">' +
    '<a href="#" data-open-waitlist>Join the waitlist</a>' +
    '<a href="#for-both" data-scroll-link>See the two sides of care</a>' +
    "</div></div>" +
    '<p class="footer-quote">Care, made more human.</p>' +
    "</footer>"
  );
}

function mirrorVisual(options) {
  const settings = options || {};
  const screen = mirrorScreens[settings.screen || "idle"];
  const size = settings.size ? " mirror-device--" + settings.size : "";
  const className = settings.className ? " " + settings.className : "";
  return (
    '<div class="product-stage mirror-stage' +
    className +
    '">' +
    '<div class="mirror-device' +
    size +
    '">' +
    '<div class="mirror-screen">' +
    '<img src="' +
    ASSET(screen.source) +
    '" alt="' +
    screen.alt +
    '" loading="lazy" />' +
    '<span class="mirror-light mirror-light--left"></span>' +
    '<span class="mirror-light mirror-light--right"></span>' +
    '<span class="mirror-camera"></span>' +
    "</div>" +
    "</div>" +
    "</div>"
  );
}

function phoneVisual(options) {
  const settings = options || {};
  const screen = phoneScreens[settings.screen || "home"];
  const size = settings.size ? " phone-device--" + settings.size : "";
  const className = settings.className ? " " + settings.className : "";
  return (
    '<div class="product-stage phone-stage' +
    className +
    '">' +
    '<div class="phone-device' +
    size +
    '">' +
    '<div class="phone-screen">' +
    '<img src="' +
    ASSET(screen.source) +
    '" alt="' +
    screen.alt +
    '" loading="lazy" />' +
    '<span class="phone-island"></span>' +
    "</div>" +
    "</div>" +
    "</div>"
  );
}

function featureList(items) {
  return (
    '<ul class="feature-list">' +
    items
      .map(
        (item) =>
          '<li><span class="icon-wrap">' +
          icon(item.icon) +
          "</span><span>" +
          item.label +
          "</span></li>",
      )
      .join("") +
    "</ul>"
  );
}

function featureStrip(items) {
  return (
    '<div class="feature-strip">' +
    items
      .map(
        (item) =>
          '<div class="feature-strip__item">' +
          '<span class="icon-wrap">' +
          icon(item.icon) +
          "</span>" +
          "<span><strong>" +
          item.title +
          "</strong><span>" +
          item.copy +
          "</span></span>" +
          "</div>",
      )
      .join("") +
    "</div>"
  );
}

function trustStrip(items, keyPrefix) {
  return (
    '<div class="trust-strip">' +
    items
      .map(
        (item, index) =>
          '<div class="trust-strip__item"' +
          (keyPrefix ? ' data-layout-key="' + keyPrefix + "." + (index + 1) + '"' : "") +
          ">" +
          '<span class="icon-wrap">' +
          icon(item.icon) +
          "</span>" +
          "<span><strong>" +
          item.title +
          "</strong><span>" +
          item.copy +
          "</span></span>" +
          "</div>",
      )
      .join("") +
    "</div>"
  );
}

function ctaBand(kicker, title, linkLabel, linkPath) {
  return (
    '<section class="section section--tight" data-reveal>' +
    '<div class="cta-band">' +
    '<span class="cta-botanical" aria-hidden="true">' +
    icon("leaf", "cta-botanical__leaf cta-botanical__leaf--one") +
    icon("leaf", "cta-botanical__leaf cta-botanical__leaf--two") +
    icon("leaf", "cta-botanical__leaf cta-botanical__leaf--three") +
    "</span>" +
    '<p class="cta-kicker">' +
    kicker +
    "</p>" +
    "<h2>" +
    title +
    "</h2>" +
    '<div class="hero-actions">' +
    button("Join the waitlist", "light") +
    routeLink(linkLabel + " " + icon("arrow"), linkPath, "text-link text-link--light") +
    "</div>" +
    "</div>" +
    "</section>"
  );
}

function homeProductAsset(kind, alt) {
  return (
    '<div class="home-product-asset home-product-asset--' +
    kind +
    '"><img src="' +
    ASSET(homeAssets[kind]) +
    '" alt="' +
    alt +
    '" loading="eager" /></div>'
  );
}

function homePage() {
  return (
    '<div class="page page--home">' +
    '<section class="hero hero--centered home-hero">' +
    '<div class="hero__copy" data-reveal>' +
    badge("Care. Connected.") +
    "<h1>Built for both sides of care.</h1>" +
    "<p>A better day for them. A clearer picture for you.</p>" +
    '<div class="hero-actions hero-actions--centered">' +
    button("Join the waitlist") +
    "</div>" +
    "</section>" +
    '<section class="section section--tight home-care-section" id="for-both">' +
    '<div class="split-care home-split-care">' +
    '<article class="care-panel care-panel--warm home-care-panel home-care-panel--mirror">' +
    '<p class="panel-label">FOR THEM</p>' +
    "<h3>A companion that feels natural.</h3>" +
    featureList([
      { icon: "brain", label: "Stay mentally engaged" },
      { icon: "link", label: "Remember routines & plans" },
      { icon: "message", label: "Hear family messages easily" },
    ]) +
    homeProductAsset("mirror", "Reflexion Mirror on a tabletop with the morning check-in screen") +
    "</article>" +
    '<article class="care-panel care-panel--sage home-care-panel home-care-panel--caregiver">' +
    '<p class="panel-label">FOR YOU</p>' +
    "<h3>Stay close without constantly checking.</h3>" +
    featureList([
      { icon: "chart", label: "See wellness trends" },
      { icon: "bell", label: "Get alerts when needed" },
      { icon: "message", label: "Send a message in seconds" },
    ]) +
    homeProductAsset("caregiver", "Reflexion caregiver app home screen on a phone") +
    "</article>" +
    "</div>" +
    "</section>" +
    '<section class="section section--tight home-connection-section">' +
    '<div class="section-heading">' +
    "<h2>One conversation connects both.</h2>" +
    "<p>Your loved one experiences a friendly daily interaction.<br />You get only the useful, respectful information that helps you stay involved.</p>" +
    "</div>" +
    featureStrip([
      { icon: "brain", title: "Engage", copy: "Gentle daily conversations" },
      { icon: "people", title: "Connect", copy: "Share messages that matter" },
      { icon: "chart", title: "Understand", copy: "See patterns & wellness trends" },
      { icon: "bell", title: "Know when", copy: "Alerts for meaningful changes" },
      { icon: "message", title: "Reach out", copy: "Communicate easily from anywhere" },
    ]) +
    "</section>" +
    ctaBand("Care. Connected.", "See what Reflexion could look like in your family.", "How it works", "/how-it-works") +
    "</div>"
  );
}

function processStep(number, tone, title, copy, content) {
  return (
    '<article class="process-step process-step--' +
    tone +
    '" data-reveal>' +
    '<span class="step-number">' +
    number +
    "</span>" +
    "<h3>" +
    title +
    "</h3>" +
    "<p>" +
    copy +
    "</p>" +
    content +
    "</article>"
  );
}

function processProductAsset(kind, alt) {
  const source = kind === "mirror" ? homeAssets.mirror : homeAssets.caregiver;
  return (
    '<div class="process-product-asset process-product-asset--' +
    kind +
    '"><img src="' +
    ASSET(source) +
    '" alt="' +
    alt +
    '" loading="eager" /></div>'
  );
}

function processBotanical() {
  return (
    '<svg class="process-botanical" viewBox="0 0 250 330" fill="none" aria-hidden="true">' +
    '<path class="process-botanical__stem" d="M91 333C110 271 120 208 143 149 163 98 190 49 225 4"/>' +
    '<path class="process-botanical__branch" d="M128 193C99 168 71 153 38 145M148 148C119 118 96 95 68 73M165 111C181 86 198 62 218 43"/>' +
    '<path class="process-botanical__leaf" d="M42 145C51 116 76 108 102 123 87 147 65 156 42 145Z"/>' +
    '<path class="process-botanical__leaf" d="M69 73C80 47 107 40 130 58 113 80 90 85 69 73Z"/>' +
    '<path class="process-botanical__leaf" d="M101 264C75 250 69 226 81 204 105 217 113 240 101 264Z"/>' +
    '<path class="process-botanical__leaf" d="M143 149C160 125 186 119 208 133 193 155 168 162 143 149Z"/>' +
    '<path class="process-botanical__leaf" d="M177 93C190 68 215 58 237 70 223 94 201 103 177 93Z"/>' +
    '<circle class="process-botanical__berry" cx="219" cy="42" r="7"/><circle class="process-botanical__berry" cx="230" cy="20" r="6"/><circle class="process-botanical__berry" cx="201" cy="29" r="5"/>' +
    "</svg>"
  );
}

function analysisList() {
  const items = [
    { icon: "message", title: "Conversation summaries", copy: "Key moments and topics from each interaction." },
    { icon: "chart", title: "Wellness trends", copy: "Mood, sleep, activity, and engagement over time." },
    { icon: "bell", title: "Smart reminders", copy: "Medication, appointments, and daily routines." },
    { icon: "people", title: "Interaction patterns", copy: "Frequency, duration, and what brings joy." },
  ];
  return (
    '<div class="analysis-list">' +
    items
      .map(
        (item) =>
          '<div class="analysis-item"><span class="icon-wrap">' +
          icon(item.icon) +
          "</span><span><strong>" +
          item.title +
          "</strong><span>" +
          item.copy +
          "</span></span></div>",
      )
      .join("") +
    "</div>"
  );
}

function processPage() {
  return (
    '<div class="page page--process">' +
    '<section class="hero hero--centered">' +
    '<div class="hero__copy" data-reveal>' +
    badge("Care. Connected.") +
    "<h1>How Reflexion works.</h1>" +
    "<p>A simple daily rhythm that supports your loved one and keeps you meaningfully connected.</p>" +
    '<div class="hero-actions hero-actions--centered">' +
    button("Join the waitlist") +
    "</div>" +
    "</div>" +
    processBotanical() +
    "</section>" +
    '<section class="section section--tight">' +
    '<div class="process-rail">' +
    processStep(
      "1",
      "warm",
      "They talk naturally",
      "Your loved one has a warm daily conversation and routine check-in with Reflexion.",
      processProductAsset("mirror", "Reflexion Mirror on a tabletop"),
    ) +
    processStep(
      "2",
      "sage",
      "Reflexion understands patterns",
      "Aria analyzes conversations and routines to uncover insights that matter most.",
      analysisList(),
    ) +
    processStep(
      "3",
      "warm",
      "You stay connected",
      "You receive meaningful updates and suggested next steps—anytime, anywhere.",
      processProductAsset("caregiver", "Reflexion caregiver app home screen"),
    ) +
    "</div>" +
    "</section>" +
    '<section class="section section--tight" data-reveal>' +
    '<div class="section-heading">' +
    "<h2>What Aria helps with</h2>" +
    "</div>" +
    '<div class="helps-grid">' +
    [
      { icon: "brain", title: "Recall & reminiscence", lines: ["Meaningful conversations", "that spark memories and", "bring comfort."] },
      { icon: "calendar", title: "Daily routines", lines: ["Gentle support to stay", "on track with what", "matters most."] },
      { icon: "message", title: "Family messages", lines: ["Share updates, photos,", "and love—delivered", "in the moment."] },
      { icon: "bell", title: "Gentle prompts", lines: ["Kind reminders that", "encourage wellness", "without overwhelming."] },
    ]
      .map(
        (item) =>
          '<div class="help-item"><span class="icon-wrap">' +
          icon(item.icon) +
          "</span><strong>" +
          item.title +
          "</strong><span>" +
          item.lines.join(' <br class="process-desktop-break" />') +
          "</span></div>",
      )
      .join("") +
    "</div>" +
    "</section>" +
    ctaBand("Care. Connected.", "Better conversations. Stronger connections.", "See the mirror", "/the-mirror") +
    "</div>"
  );
}

function familyFeature(iconName, title, copy) {
  return (
    '<article class="family-feature" data-reveal>' +
    '<span class="icon-wrap">' +
    icon(iconName) +
    "</span><h3>" +
    title +
    "</h3><p>" +
    copy +
    "</p></article>"
  );
}

function familiesCaregiverAsset() {
  return (
    '<div class="families-phone-asset"><img src="' +
    ASSET(homeAssets.caregiver) +
    '" alt="Reflexion caregiver app home screen" loading="eager" /></div>'
  );
}

function familyFeatureCard(iconName, title, copy, variant) {
  let preview = "";

  if (variant === "trends") {
    preview =
      '<svg class="family-trend-graph" viewBox="0 0 240 92" fill="none" aria-hidden="true">' +
      '<path class="family-trend-graph__line" d="M2 75C17 66 24 65 38 68 53 72 59 57 74 56 89 55 95 42 108 46 122 51 129 69 143 71 158 73 168 54 181 43 194 31 203 44 215 45 225 46 230 32 238 19"/>' +
      '<circle cx="38" cy="68" r="4"/><circle cx="74" cy="56" r="4"/><circle cx="108" cy="46" r="4"/><circle cx="143" cy="71" r="4"/><circle cx="181" cy="43" r="4"/><circle cx="215" cy="45" r="4"/><circle cx="238" cy="19" r="4"/>' +
      "</svg>";
  }

  if (variant === "alerts") {
    preview =
      '<div class="family-alert-preview">' +
      '<span class="family-avatar family-avatar--dad"></span>' +
      '<span><strong>Dad Robert</strong><small>Needs your attention</small></span>' +
      '<b>9:20 AM</b><i></i>' +
      "</div>";
  }

  if (variant === "messages") {
    preview =
      '<div class="family-message-preview">' +
      '<span class="family-avatar family-avatar--mum"></span>' +
      '<span><strong>Mum Mary</strong><small>Morning check-in completed</small></span>' +
      '<b>8:15 AM</b><i></i>' +
      "</div>";
  }

  if (variant === "summary") {
    preview =
      '<div class="family-summary-preview"><span class="family-summary-preview__icon">' +
      icon("document") +
      '</span><p>Mum was feeling good and enjoyed her morning walk.<br />She mentioned lunch with her sister on Sunday.</p></div>';
  }

  return (
    '<article class="family-feature family-feature--' +
    variant +
    '" data-reveal>' +
    '<span class="icon-wrap">' +
    icon(iconName) +
    "</span><h3>" +
    title +
    "</h3><p>" +
    copy +
    "</p>" +
    preview +
    "</article>"
  );
}

function familiesPage() {
  return (
    '<div class="page page--families">' +
    '<section class="hero hero--centered families-hero">' +
    '<div class="hero__copy" data-reveal>' +
    badge("For families") +
    "<h1>Designed for families<br />who want to stay close.</h1>" +
    "<p>Get a clearer picture of daily life without<br />hovering, guessing, or constantly checking in.</p>" +
    '<div class="hero-actions hero-actions--centered">' +
    button("Join the waitlist") +
    "</div>" +
    "</div>" +
    '<div class="families-hero-grid">' +
    '<div class="family-feature-column family-feature-column--left">' +
    familyFeatureCard("chart", "See wellness trends", "Track patterns over time to understand how your loved one is doing.", "trends") +
    familyFeatureCard("bell", "Get alerts when needed", "Smart notifications keep you informed about what matters most.", "alerts") +
    "</div>" +
    '<div class="families-phone" data-reveal>' +
    familiesCaregiverAsset() +
    "</div>" +
    '<div class="family-feature-column family-feature-column--right">' +
    familyFeatureCard("message", "Call or message in seconds", "Reach out quickly with one tap—no logins, no waiting.", "messages") +
    familyFeatureCard("shield", "Stay grounded in real conversation summaries", "See what was discussed so you never miss the meaningful moments.", "summary") +
    "</div>" +
    "</div>" +
    '<svg class="families-botanical families-botanical--right" viewBox="0 0 170 250" fill="none" aria-hidden="true">' +
    '<path d="M42 249C54 192 77 131 124 72 137 56 151 39 164 18"/>' +
    '<path d="M77 154C54 132 36 117 10 108M101 115C78 89 61 68 43 39M124 73C139 58 151 43 160 28"/>' +
    '<path d="M11 108C18 84 40 76 61 88 48 108 29 115 11 108ZM43 39C54 18 77 12 96 27 82 46 60 51 43 39ZM77 154C53 148 42 130 47 112 67 118 80 135 77 154ZM101 115C113 93 135 88 151 99 140 117 119 123 101 115ZM124 73C139 52 159 51 170 61 159 78 141 82 124 73Z"/>' +
    "</svg>" +
    '<svg class="families-botanical families-botanical--left" viewBox="0 0 120 250" fill="none" aria-hidden="true">' +
    '<path d="M28 250C34 186 51 121 92 65"/>' +
    '<path d="M48 181C30 163 17 145 3 121M62 132C45 111 32 90 21 62M81 85C91 68 101 52 111 37"/>' +
    '<path d="M3 121C9 101 27 94 44 103 34 121 18 128 3 121ZM21 62C31 42 51 37 68 49 56 67 38 72 21 62ZM48 181C29 175 20 159 24 144 40 149 51 164 48 181ZM62 132C73 112 92 108 106 119 96 135 78 140 62 132ZM81 85C94 65 111 65 120 74 110 91 95 95 81 85Z"/>' +
    "</svg>" +
    "</section>" +
    '<section class="section section--tight families-lower" data-reveal>' +
    '<figure class="testimonial"><blockquote>Reflexion gives me peace of mind<br />without making Mum feel like<br />I’m watching over her.</blockquote><figcaption><span class="testimonial-avatar" aria-hidden="true"></span><span>— Emma, daughter of Mary</span></figcaption></figure>' +
    '<div class="insight-callout families-insight">' +
    '<span class="icon-wrap">' +
    icon("leaf") +
    "</span>" +
    "<span><strong>Suggested next step</strong><h3>See how Reflexion keeps you<br />connected with what matters.</h3><p>A quick call or message can make their day.</p></span>" +
    '<span class="families-insight-actions">' +
    button("Join the waitlist", null, "button--small") +
    routeLink("See how it works " + icon("arrow"), "/how-it-works", "text-link") +
    "</span>" +
    '<svg class="families-insight-botanical" viewBox="0 0 170 190" fill="none" aria-hidden="true">' +
    '<path d="M45 190C56 143 79 94 125 45 139 30 151 18 164 4"/>' +
    '<path d="M78 126C54 108 37 94 12 87M101 89C78 69 60 48 44 28M126 45C140 31 151 19 160 10"/>' +
    '<path d="M12 87C19 68 40 61 59 72 47 89 29 94 12 87ZM44 28C55 10 76 5 94 18 81 35 61 39 44 28ZM78 126C56 120 46 105 50 90 68 96 81 111 78 126ZM101 89C113 70 133 66 149 76 139 93 119 98 101 89ZM126 45C139 27 158 27 170 37 159 53 141 56 126 45Z"/>' +
    "</svg>" +
    "</div>" +
    "</section>" +
    '<section class="section section--tight families-trust" data-reveal>' +
    '<div class="section-heading">' +
    "<h2>Why families love Reflexion</h2>" +
    "</div>" +
    trustStrip([
      { icon: "heart", title: "Useful updates", copy: "Timely insights help you support without guessing." },
      { icon: "shield", title: "Respectful insights", copy: "Private by design, so independence is protected." },
      { icon: "people", title: "Connection made easy", copy: "Simple tools help you stay close, every day." },
    ]) +
    "</section>" +
    "</div>"
  );
}

function callout(iconName, title, copy) {
  return (
    '<article class="callout" data-reveal><div class="callout__heading"><span class="icon-wrap">' +
    icon(iconName) +
    "</span><strong>" +
    title +
    "</strong></div><p>" +
    copy +
    "</p></article>"
  );
}

function mirrorFeatureCard(imageName, imageAlt, title, copy) {
  return (
    '<article class="mirror-feature-card" data-reveal><div class="mirror-feature-card__image"><img src="' +
    ASSET(imageName) +
    '" alt="' +
    imageAlt +
    '" loading="lazy" /></div><div class="mirror-feature-card__body"><h3>' +
    title +
    "</h3><p>" +
    copy +
    "</p></div></article>"
  );
}

function mirrorHeroCallout(side, iconName, title, copy) {
  return (
    '<article class="mirror-hero-callout mirror-hero-callout--' +
    side +
    '" data-reveal><span class="mirror-hero-callout__icon">' +
    icon(iconName) +
    '</span><div><h3>' +
    title +
    '</h3><p>' +
    copy +
    '</p></div></article>'
  );
}

function mirrorEverydayCard(variant, title, copy) {
  return (
    '<article class="mirror-everyday-card mirror-everyday-card--' +
    variant +
    '" data-reveal><div class="mirror-everyday-card__image"><img src="' +
    ASSET(homeAssets.mirror) +
    '" alt="Reflexion Mirror in a calm home setting" loading="eager" /></div><div class="mirror-everyday-card__body"><span class="icon-wrap">' +
    icon(variant === "older" ? "heart" : variant === "home" ? "home" : "wave") +
    '</span><h3>' +
    title +
    '</h3><p>' +
    copy +
    '</p></div></article>'
  );
}

function mirrorPage() {
  return (
    '<div class="page page--mirror">' +
    '<section class="mirror-hero">' +
    '<div class="mirror-hero__copy" data-reveal>' +
    badge("Care. Connected.") +
    "<h1>A familiar mirror, reimagined for gentle daily care.</h1>" +
    "<p>Voice-first, visually calm, and<br />designed to feel natural in the home.</p>" +
    '<div class="hero-actions">' +
    button("Join the waitlist") +
    "</div>" +
    "</div>" +
    '<div class="mirror-hero__scene">' +
    '<svg class="mirror-hero__botanical" viewBox="0 0 220 340" fill="none" aria-hidden="true">' +
    '<path d="M60 339C65 276 85 196 132 122 151 92 174 58 202 20"/><path d="M84 251C55 231 31 212 10 184M106 193C80 171 63 145 46 113M132 122C151 96 171 67 185 43"/>' +
    '<path d="M10 184C16 156 40 144 63 157 50 182 29 192 10 184ZM46 113C58 87 83 78 105 94 90 119 66 126 46 113ZM84 251C57 244 43 224 49 203 72 211 88 230 84 251ZM106 193C121 165 147 156 167 171 153 197 128 204 106 193ZM132 122C150 93 179 87 198 102 184 130 155 138 132 122Z"/>' +
    '</svg>' +
    '<div class="mirror-hero__asset"><img src="' +
    ASSET(homeAssets.mirror) +
    '" alt="Reflexion Mirror showing a morning check-in on a tabletop" loading="eager" /></div>' +
    '<div class="mirror-hero__callouts mirror-hero__callouts--left">' +
    mirrorHeroCallout("left", "wave", "Voice-first", "Just speak naturally. The mirror listens and responds.") +
    mirrorHeroCallout("left", "check", "Simple morning check-in", "A gentle way to start the day.") +
    mirrorHeroCallout("left", "message", "Family messages", "See kind notes and updates from the people who care.") +
    '</div>' +
    '<div class="mirror-hero__callouts mirror-hero__callouts--right">' +
    mirrorHeroCallout("right", "calendar", "Daily routine support", "Stay on track with reminders that feel helpful, not intrusive.") +
    mirrorHeroCallout("right", "sun", "Ambient side lights", "Soft, adjustable lighting that’s easy on the eyes.") +
    mirrorHeroCallout("right", "home", "Thoughtful tabletop design", "Sleek, stable, and made to feel at home.") +
    '</div>' +
    "</div>" +
    "</section>" +
    '<section class="mirror-everyday" data-reveal>' +
    '<div class="section-heading">' +
    "<h2>Thoughtfully designed for everyday life.</h2>" +
    "<p>No screens to figure out. No apps to open. Just a better way to stay connected and supported.</p>" +
    "</div>" +
    '<div class="mirror-everyday-grid">' +
    mirrorEverydayCard("learn", "No app to learn", "Everything happens right on the mirror. Just step up and start talking.") +
    mirrorEverydayCard("older", "Designed for older adults", "Large text, clear audio, and simple language make it easy and comfortable.") +
    mirrorEverydayCard("home", "Built for the home", "Beautiful, minimal, and calming—it fits right in.") +
    "</div>" +
    "</section>" +
    ctaBand("Care. Connected.", "See what Reflexion could look like in your home.", "How it works", "/how-it-works") +
    "</div>"
  );
}

function principleCard(iconName, title, copy, layoutKey) {
  return (
    '<article class="principle-card" data-reveal' +
    (layoutKey ? ' data-layout-key="' + layoutKey + '"' : "") +
    '><span class="icon-wrap">' +
    icon(iconName) +
    "</span><h3>" +
    title +
    "</h3><p>" +
    copy +
    "</p></article>"
  );
}

function aboutBotanical(className, layoutKey) {
  return (
    '<svg class="about-botanical ' +
    className +
    '"' +
    (layoutKey ? ' data-layout-key="' + layoutKey + '"' : "") +
    ' viewBox="0 0 220 340" fill="none" aria-hidden="true">' +
    '<path d="M60 339C65 276 85 196 132 122 151 92 174 58 202 20"/><path d="M84 251C55 231 31 212 10 184M106 193C80 171 63 145 46 113M132 122C151 96 171 67 185 43"/>' +
    '<path d="M10 184C16 156 40 144 63 157 50 182 29 192 10 184ZM46 113C58 87 83 78 105 94 90 119 66 126 46 113ZM84 251C57 244 43 224 49 203 72 211 88 230 84 251ZM106 193C121 165 147 156 167 171 153 197 128 204 106 193ZM132 122C150 93 179 87 198 102 184 130 155 138 132 122Z"/>' +
    '</svg>'
  );
}

function aboutPage() {
  return (
    '<div class="page page--about">' +
    '<section class="about-hero">' +
    '<div class="about-hero__copy" data-reveal data-layout-key="hero.copy">' +
    badge("Our mission") +
    "<h1>Built on trust.<br />Designed for dignity.</h1>" +
    "<p>Reflexion is designed to help families stay connected<br />through respectful conversation, useful context,<br />and thoughtful design.</p>" +
    '<div class="hero-actions">' +
    button("Join the waitlist") +
    routeLink("See how it works " + icon("arrow"), "/how-it-works", "text-link") +
    "</div>" +
    "</div>" +
    '<div class="about-hero__scene">' +
    '<div class="about-hero__mirror" data-layout-key="hero.mirror"><img src="' +
    ASSET(homeAssets.mirror) +
    '" alt="Reflexion Mirror showing a morning check-in" loading="eager" /></div>' +
    '<div class="about-hero__phone" data-layout-key="hero.phone"><img src="' +
    ASSET(homeAssets.caregiver) +
    '" alt="Reflexion caregiver app home screen" loading="eager" /></div>' +
    aboutBotanical("about-botanical--hero", "hero.botanical") +
    '<div class="about-hero__vase" data-layout-key="hero.vase" aria-hidden="true"></div><div class="about-hero__books" data-layout-key="hero.books" aria-hidden="true"></div>' +
    "</div>" +
    "</section>" +
    '<section class="about-principles" data-reveal>' +
    '<div class="principles-grid">' +
    principleCard("people", "Supports connection, not surveillance", "Reflexion is here to spark meaningful conversations, not track or monitor.", "principle.1") +
    principleCard("bell", "Flags changes, not diagnoses", "We highlight patterns that may matter—so families can check in and respond.", "principle.2") +
    principleCard("shield", "Built for privacy and consent", "Personal data stays private and is shared only with your permission.", "principle.3") +
    principleCard("heart", "Designed for real families", "Every detail—from the mirror to the app—was shaped by caregivers like you.", "principle.4") +
    "</div>" +
    "</section>" +
    '<section class="about-story-section" data-reveal>' +
    '<div class="about-story">' +
    '<div class="about-story__copy" data-layout-key="story.copy"><p class="panel-label">OUR STORY</p><h2>Born from a simple<br />but powerful belief.</h2><p>Reflexion was created by families who know the quiet<br />worry of living with someone from afar.</p><p>We believe that a daily check-in can be a bridge—bringing<br />reassurance, preserving independence, and strengthening<br />the bonds that matter most.</p><p class="story-signoff">With care,<span>The Reflexion Team</span></p></div>' +
    '<div class="about-story__scene"><div class="about-story__mirror" data-layout-key="story.mirror"><img src="' +
    ASSET(homeAssets.mirror) +
    '" alt="Reflexion Mirror in the home" loading="eager" /></div><div class="about-story__phone" data-layout-key="story.phone"><img src="' +
    ASSET(homeAssets.caregiver) +
    '" alt="Reflexion caregiver app home screen" loading="eager" /></div>' +
    aboutBotanical("about-botanical--story", "story.botanical") +
    "</div>" +
    "</div>" +
    "</section>" +
    '<section class="about-trust-section" data-reveal><div class="about-trust">' +
    '<div class="about-trust__heading"><h2>Trust is at the heart of everything we build.</h2></div>' +
    trustStrip([
      { icon: "shield", title: "Privacy first", copy: "Your data is private, secure, and never sold." },
      { icon: "heart", title: "Dignity always", copy: "We design with respect for independence." },
      { icon: "people", title: "Family-centered", copy: "Tools that empower caregivers and loved ones." },
      { icon: "check", title: "Real-life ready", copy: "Simple, warm, and built for everyday use." },
    ], "trust") +
    "</div></section>" +
    "</div>"
  );
}

function pageContent() {
  const path = currentPath();
  if (path === "/how-it-works") return processPage();
  if (path === "/for-families") return familiesPage();
  if (path === "/the-mirror") return mirrorPage();
  if (path === "/about") return aboutPage();
  return homePage();
}

function modalMarkup() {
  return (
    '<div class="modal-backdrop" id="waitlist-modal" role="presentation" hidden>' +
    '<section class="modal" role="dialog" aria-modal="true" aria-labelledby="waitlist-title">' +
    '<button type="button" class="modal-close" data-close-waitlist aria-label="Close waitlist form">' +
    icon("close") +
    "</button>" +
    badge("Local demo") +
    '<div id="waitlist-content">' +
    '<h2 id="waitlist-title">Stay close to what comes next.</h2>' +
    "<p>This demo form is a placeholder for the future Reflexion waitlist. Nothing is submitted anywhere yet.</p>" +
    '<form class="waitlist-form">' +
    '<label for="waitlist-email">Email address</label>' +
    '<input id="waitlist-email" name="email" type="email" placeholder="you@example.com" autocomplete="email" required />' +
    '<p class="form-error" aria-live="polite"></p>' +
    formButton("Save my place") +
    "</form>" +
    "</div>" +
    "</section>" +
    "</div>"
  );
}

const layoutEditorStorageKey = "reflexion-marketing-layout-about-v1";

function layoutEditorEnabled() {
  return currentPath() === "/about" && new URLSearchParams(window.location.search).get("edit") === "1";
}

function layoutNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.round(numeric) : 0;
}

function readLayoutState() {
  try {
    const stored = window.localStorage.getItem(layoutEditorStorageKey);
    const parsed = stored ? JSON.parse(stored) : {};
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function writeLayoutState(state) {
  try {
    window.localStorage.setItem(layoutEditorStorageKey, JSON.stringify(state));
  } catch {
    // The editor still works for the current session if storage is unavailable.
  }
}

function applyLayoutOffsets(state) {
  if (currentPath() !== "/about") return;
  const offsets = state || readLayoutState();
  document.querySelectorAll("[data-layout-key]").forEach((target) => {
    const saved = offsets[target.dataset.layoutKey] || {};
    target.classList.add("layout-editable");
    target.style.setProperty("--layout-x", layoutNumber(saved.x) + "px");
    target.style.setProperty("--layout-y", layoutNumber(saved.y) + "px");
  });
}

function layoutTargetLabel(target) {
  const key = target.dataset.layoutKey || "element";
  const parts = key.split(".");
  const last = parts[parts.length - 1];
  const readable = last.replace(/^(\d+)$/, "card $1").replace(/-/g, " ");
  return parts.length > 1 ? parts[0] + " / " + readable : readable;
}

function layoutEditorMarkup() {
  return (
    '<aside class="layout-editor" data-layout-editor aria-label="About layout editor">' +
    '<div class="layout-editor__intro"><span class="layout-editor__icon">' +
    icon("spark") +
    '</span><span><strong>Edit layout</strong><small>Drag the outlined pieces into place.</small></span></div>' +
    '<div class="layout-editor__actions"><button type="button" data-layout-action="reset">Reset</button><button type="button" data-layout-action="close">Done</button></div>' +
    '<p class="layout-editor__status" data-layout-status>Positions save in this browser.</p>' +
    '<p class="layout-editor__hint">Tip: press E on About any time to reopen this editor.</p>' +
    "</aside>"
  );
}

function openLayoutEditor() {
  const url = new URL(window.location.href);
  url.searchParams.set("edit", "1");
  window.history.replaceState({}, "", url.pathname + url.search + url.hash);
  render();
}

function closeLayoutEditor() {
  const url = new URL(window.location.href);
  url.searchParams.delete("edit");
  window.history.replaceState({}, "", url.pathname + url.search + url.hash);
  render();
}

function resetLayoutOffsets() {
  writeLayoutState({});
  applyLayoutOffsets({});
  const status = document.querySelector("[data-layout-status]");
  if (status) status.textContent = "Positions reset to the reference layout.";
}

function initLayoutEditor() {
  const targets = Array.from(document.querySelectorAll("[data-layout-key]"));
  const status = document.querySelector("[data-layout-status]");
  const setStatus = (message) => {
    if (status) status.textContent = message;
  };

  targets.forEach((target) => {
    const label = layoutTargetLabel(target);
    target.dataset.layoutLabel = label;
    target.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      const eventTarget = event.target;
      if (eventTarget instanceof Element && eventTarget.closest("a, button, input, textarea, select")) return;
      event.preventDefault();

      const saved = readLayoutState()[target.dataset.layoutKey] || {};
      const startOffset = { x: layoutNumber(saved.x), y: layoutNumber(saved.y) };
      const startPoint = { x: event.clientX, y: event.clientY };
      let currentOffset = { ...startOffset };

      target.dataset.layoutDragging = "true";
      target.style.zIndex = "80";
      try {
        if (target.setPointerCapture) target.setPointerCapture(event.pointerId);
      } catch {
        // Synthetic pointer events used by local QA do not always have a capturable pointer.
      }
      setStatus("Moving " + label + "…");

      const onMove = (moveEvent) => {
        currentOffset = {
          x: startOffset.x + moveEvent.clientX - startPoint.x,
          y: startOffset.y + moveEvent.clientY - startPoint.y,
        };
        target.style.setProperty("--layout-x", Math.round(currentOffset.x) + "px");
        target.style.setProperty("--layout-y", Math.round(currentOffset.y) + "px");
        setStatus(
          label +
            " · " +
            (currentOffset.x >= 0 ? "+" : "") +
            Math.round(currentOffset.x) +
            "px, " +
            (currentOffset.y >= 0 ? "+" : "") +
            Math.round(currentOffset.y) +
            "px",
        );
      };

      const finish = () => {
        const nextState = readLayoutState();
        nextState[target.dataset.layoutKey] = {
          x: Math.round(currentOffset.x),
          y: Math.round(currentOffset.y),
        };
        writeLayoutState(nextState);
        target.removeEventListener("pointermove", onMove);
        target.removeEventListener("pointerup", finish);
        target.removeEventListener("pointercancel", finish);
        target.removeAttribute("data-layout-dragging");
        target.style.zIndex = "";
        try {
          if (target.releasePointerCapture) target.releasePointerCapture(event.pointerId);
        } catch {
          // See the setPointerCapture note above.
        }
        setStatus(label + " saved locally.");
      };

      target.addEventListener("pointermove", onMove);
      target.addEventListener("pointerup", finish);
      target.addEventListener("pointercancel", finish);
    });
  });

  setStatus("Drag an outlined element. Positions save automatically.");
}

function render() {
  const existingLayoutEditor = document.querySelector("[data-layout-editor]");
  if (existingLayoutEditor) existingLayoutEditor.remove();
  document.body.classList.remove("is-layout-editing");
  const app = document.querySelector("#app");
  const homeClass = currentPath() === "/" ? " site-shell--home" : "";
  app.innerHTML =
    '<div class="site-shell' +
    homeClass +
    '">' +
    header() +
    '<main class="main">' +
    pageContent() +
    "</main>" +
    footer() +
    "</div>" +
    modalMarkup();
  initMotion();
  initHeaderState();
  applyLayoutOffsets();
  if (layoutEditorEnabled()) {
    document.body.classList.add("is-layout-editing");
    document.body.insertAdjacentHTML("beforeend", layoutEditorMarkup());
    initLayoutEditor();
  }
}

function openWaitlist() {
  const modal = document.querySelector("#waitlist-modal");
  if (!modal) return;
  modal.hidden = false;
  document.body.classList.add("modal-open");
  const input = modal.querySelector("input");
  if (input) window.setTimeout(() => input.focus(), 40);
}

function closeWaitlist() {
  const modal = document.querySelector("#waitlist-modal");
  if (!modal) return;
  modal.hidden = true;
  document.body.classList.remove("modal-open");
}

function showWaitlistSuccess() {
  const content = document.querySelector("#waitlist-content");
  if (!content) return;
  content.innerHTML =
    '<div class="modal-success">' +
    '<span class="icon-wrap">' +
    icon("check") +
    "</span>" +
    "<h3>You’re on the local list.</h3>" +
    "<p>Thanks for exploring the demo. The real waitlist connection can be added later without changing this experience.</p>" +
    '<button type="button" class="button button--outline button--small" data-close-waitlist>Close</button>' +
    "</div>";
}

function initMotion() {
  const targets = document.querySelectorAll("[data-reveal]");
  if (!("IntersectionObserver" in window)) {
    targets.forEach((target) => target.classList.add("is-visible"));
    return;
  }
  const observer = new IntersectionObserver(
    (entries, instance) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          instance.unobserve(entry.target);
        }
      });
    },
    { rootMargin: "0px 0px -9% 0px", threshold: 0.08 },
  );
  targets.forEach((target) => observer.observe(target));
}

function initHeaderState() {
  const headerElement = document.querySelector(".site-header");
  if (!headerElement) return;
  const setScrolled = () => headerElement.classList.toggle("is-scrolled", window.scrollY > 8);
  setScrolled();
  window.addEventListener("scroll", setScrolled, { passive: true });
}

document.addEventListener("click", (event) => {
  const layoutAction = event.target.closest("[data-layout-action]");
  if (layoutAction) {
    event.preventDefault();
    if (layoutAction.dataset.layoutAction === "reset") resetLayoutOffsets();
    if (layoutAction.dataset.layoutAction === "close") closeLayoutEditor();
    return;
  }

  const route = event.target.closest("[data-route]");
  if (route) {
    const href = route.getAttribute("href");
    if (href && href.startsWith("/")) {
      event.preventDefault();
      document.body.classList.remove("nav-open");
      window.history.pushState({}, "", href);
      render();
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
    return;
  }

  if (event.target.closest("[data-open-waitlist]")) {
    event.preventDefault();
    openWaitlist();
    return;
  }

  if (event.target.closest("[data-close-waitlist]") || event.target.id === "waitlist-modal") {
    closeWaitlist();
    return;
  }

  const navToggle = event.target.closest(".nav-toggle");
  if (navToggle) {
    const isOpen = document.body.classList.toggle("nav-open");
    navToggle.setAttribute("aria-expanded", String(isOpen));
  }
});

document.addEventListener("submit", (event) => {
  if (!event.target.matches(".waitlist-form")) return;
  event.preventDefault();
  const form = event.target;
  const input = form.querySelector("input");
  const error = form.querySelector(".form-error");
  if (!input.value.trim() || !input.checkValidity()) {
    error.textContent = "Enter a valid email address to continue.";
    input.focus();
    return;
  }
  showWaitlistSuccess();
});

document.addEventListener("keydown", (event) => {
  const tagName = event.target && event.target.tagName;
  if (
    event.key.toLowerCase() === "e" &&
    currentPath() === "/about" &&
    !layoutEditorEnabled() &&
    !event.metaKey &&
    !event.ctrlKey &&
    !event.altKey &&
    !["INPUT", "TEXTAREA", "SELECT"].includes(tagName)
  ) {
    event.preventDefault();
    openLayoutEditor();
    return;
  }

  if (event.key === "Escape") closeWaitlist();
});

window.addEventListener("popstate", render);

render();
