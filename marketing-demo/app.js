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

function trustStrip(items) {
  return (
    '<div class="trust-strip">' +
    items
      .map(
        (item) =>
          '<div class="trust-strip__item">' +
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
    "</section>" +
    '<section class="section section--tight">' +
    '<div class="process-rail">' +
    processStep(
      "1",
      "warm",
      "They talk naturally",
      "Your loved one has a warm daily conversation and routine check-in with Reflexion.",
      mirrorVisual({ screen: "speaking" }),
    ) +
    processStep(
      "2",
      "sage",
      "Reflexion understands patterns",
      "Moments are organised into useful, respectful context for the people who care.",
      analysisList(),
    ) +
    processStep(
      "3",
      "warm",
      "You stay connected",
      "You receive meaningful updates and suggested next steps—anytime, anywhere.",
      phoneVisual({ screen: "home" }),
    ) +
    "</div>" +
    "</section>" +
    '<section class="section section--tight" data-reveal>' +
    '<div class="section-heading">' +
    "<h2>What Aria helps with</h2>" +
    "<p>Small moments of support, made easier to find and share.</p>" +
    "</div>" +
    '<div class="helps-grid">' +
    [
      { icon: "brain", title: "Recall & reminiscence", copy: "Meaningful conversations that spark memories and bring comfort." },
      { icon: "calendar", title: "Daily routines", copy: "Gentle support to stay on track with what matters most." },
      { icon: "message", title: "Family messages", copy: "Share updates, photos, and love—delivered in the moment." },
      { icon: "bell", title: "Gentle prompts", copy: "Kind reminders that encourage wellness without overwhelming." },
    ]
      .map(
        (item) =>
          '<div class="help-item"><span class="icon-wrap">' +
          icon(item.icon) +
          "</span><strong>" +
          item.title +
          "</strong><span>" +
          item.copy +
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

function familiesPage() {
  return (
    '<div class="page page--families">' +
    '<section class="hero hero--centered">' +
    '<div class="hero__copy" data-reveal>' +
    badge("For families") +
    "<h1>Designed for families who want to stay close.</h1>" +
    "<p>Get a clearer picture of daily life without hovering, guessing, or constantly checking in.</p>" +
    '<div class="hero-actions hero-actions--centered">' +
    button("Join the waitlist") +
    "</div>" +
    "</div>" +
    '<div class="families-hero-grid">' +
    '<div class="family-feature-column">' +
    familyFeature("chart", "See wellness trends", "Track patterns over time to understand how your loved one is doing.") +
    familyFeature("bell", "Get alerts when needed", "Useful notifications keep you informed about what matters most.") +
    "</div>" +
    '<div class="families-phone" data-reveal>' +
    phoneVisual({ screen: "home", size: "large" }) +
    "</div>" +
    '<div class="family-feature-column">' +
    familyFeature("message", "Call or message in seconds", "Reach out quickly with one tap—no logins, no waiting.") +
    familyFeature("shield", "Stay grounded in real conversation summaries", "See what was discussed so you never miss the meaningful moments.") +
    "</div>" +
    "</div>" +
    "</section>" +
    '<section class="section section--tight" data-reveal>' +
    '<div class="insight-callout">' +
    '<span class="icon-wrap">' +
    icon("leaf") +
    "</span>" +
    "<span><strong>Suggested next step</strong><h3>See how Reflexion keeps you connected with what matters.</h3><p>A quick call or message can make their day.</p></span>" +
    button("Join the waitlist", "outline", "button--small") +
    "</div>" +
    "</section>" +
    '<section class="section section--tight" data-reveal>' +
    '<div class="section-heading">' +
    "<h2>Useful updates, right when you need them.</h2>" +
    "<p>Reflexion keeps the focus on observed moments, clear context, and respectful next steps.</p>" +
    "</div>" +
    '<div class="families-visual-cards">' +
    '<article class="visual-card">' +
    '<div class="visual-card__image"><img src="' +
    ASSET("caregiver-trends.png") +
    '" alt="Reflexion trends screen showing conversation patterns" loading="lazy" /></div>' +
    '<div class="visual-card__body"><span class="icon-wrap">' +
    icon("chart") +
    "</span><h3>See the pattern</h3><p>Follow conversation activity over time without turning care into a dashboard.</p></div>" +
    "</article>" +
    '<article class="visual-card">' +
    '<div class="visual-card__image"><img src="' +
    ASSET("caregiver-chat.png") +
    '" alt="Reflexion caregiver family messages screen" loading="lazy" /></div>' +
    '<div class="visual-card__body"><span class="icon-wrap">' +
    icon("message") +
    "</span><h3>Keep the thread</h3><p>Send something kind, see what matters, and stay part of the everyday.</p></div>" +
    "</article>" +
    '<article class="visual-card">' +
    '<div class="visual-card__image"><img src="' +
    ASSET("caregiver-routines.png") +
    '" alt="Reflexion routine setup screen" loading="lazy" /></div>' +
    '<div class="visual-card__body"><span class="icon-wrap">' +
    icon("calendar") +
    "</span><h3>Support the routine</h3><p>Make gentle prompts easier to set up and easier to follow.</p></div>" +
    "</article>" +
    "</div>" +
    "</section>" +
    '<section class="section section--tight" data-reveal>' +
    '<figure class="testimonial"><blockquote>Reflexion gives me peace of mind without making Mum feel like I’m watching over her.</blockquote><figcaption>— Emma, daughter of Mary</figcaption></figure>' +
    "</section>" +
    '<section class="section section--tight" data-reveal>' +
    '<div class="section-heading">' +
    "<h2>Why families love Reflexion</h2>" +
    "</div>" +
    trustStrip([
      { icon: "heart", title: "Useful updates", copy: "Timely insights help you support without guessing." },
      { icon: "shield", title: "Respectful insights", copy: "Private by design, so independence is protected." },
      { icon: "people", title: "Connection made easy", copy: "Simple tools help you stay close, every day." },
      { icon: "check", title: "Made for real life", copy: "A calm rhythm that fits the way families actually care." },
    ]) +
    "</section>" +
    ctaBand("Care. Connected.", "See what Reflexion could look like in your home.", "How it works", "/how-it-works") +
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

function mirrorPage() {
  return (
    '<div class="page page--mirror">' +
    '<section class="hero">' +
    '<div class="hero__grid">' +
    '<div class="hero__copy" data-reveal>' +
    badge("Care. Connected.") +
    "<h1>A familiar mirror, reimagined for gentle daily care.</h1>" +
    "<p>Voice-first, visually calm, and designed to feel natural in the home.</p>" +
    '<div class="hero-actions">' +
    button("Join the waitlist") +
    "</div>" +
    "</div>" +
    '<div class="hero__visual" data-reveal>' +
    '<div class="mirror-annotated-stage">' +
    mirrorVisual({ screen: "idle", size: "large", className: "mirror-device--hero" }) +
    "</div>" +
    "</div>" +
    "</div>" +
    "</section>" +
    '<section class="section section--tight">' +
    '<div class="annotated-layout">' +
    '<div class="callout-stack callout-stack--left">' +
    callout("wave", "Voice-first", "Just speak naturally. The mirror listens and responds.") +
    callout("check", "Simple morning check-in", "A gentle way to start the day.") +
    callout("message", "Family messages", "See kind notes and updates from the people who care.") +
    "</div>" +
    '<div class="mirror-annotated-stage" data-reveal>' +
    mirrorVisual({ screen: "idle", size: "large" }) +
    "</div>" +
    '<div class="callout-stack callout-stack--right">' +
    callout("calendar", "Daily routine support", "Stay on track with reminders that feel helpful, not intrusive.") +
    callout("sun", "Ambient side lights", "Soft, adjustable lighting that’s easy on the eyes.") +
    callout("home", "Thoughtful tabletop design", "Sleek, stable, and made to feel at home.") +
    "</div>" +
    "</div>" +
    "</section>" +
    '<section class="section section--tight" data-reveal>' +
    '<div class="section-heading">' +
    "<h2>Thoughtfully designed for everyday life.</h2>" +
    "<p>No screens to figure out. No apps to open. Just a better way to stay connected and supported.</p>" +
    "</div>" +
    '<div class="mirror-feature-grid">' +
    mirrorFeatureCard("mirror-speaking.png", "Reflexion Mirror voice interaction screen", "No app to learn", "Everything happens right on the mirror. Just step up and start talking.") +
    mirrorFeatureCard("mirror-routine.png", "Reflexion Mirror routine reminder screen", "Designed for older adults", "Large text, clear audio, and simple language make it easy and comfortable.") +
    mirrorFeatureCard("mirror-message.png", "Reflexion Mirror family message screen", "Built for the home", "Beautiful, minimal, and calming—it fits right in.") +
    "</div>" +
    "</section>" +
    '<section class="section section--tight" data-reveal>' +
    '<div class="mirror-note"><div><p class="panel-label">A calmer kind of technology</p><h2>Care that feels like part of the room.</h2><p>Reflexion is designed around ordinary moments: a morning hello, a reminder, a note from family, a little more ease.</p></div>' +
    button("Join the waitlist") +
    "</div>" +
    "</section>" +
    ctaBand("Care. Connected.", "See what Reflexion could look like in your home.", "For families", "/for-families") +
    "</div>"
  );
}

function principleCard(iconName, title, copy) {
  return (
    '<article class="principle-card" data-reveal><span class="icon-wrap">' +
    icon(iconName) +
    "</span><h3>" +
    title +
    "</h3><p>" +
    copy +
    "</p></article>"
  );
}

function aboutPage() {
  return (
    '<div class="page page--about">' +
    '<section class="hero">' +
    '<div class="hero__grid">' +
    '<div class="hero__copy" data-reveal>' +
    badge("Our mission") +
    "<h1>Built on trust. Designed for dignity.</h1>" +
    "<p>Reflexion is designed to help families stay connected through respectful conversation, useful context, and thoughtful design.</p>" +
    '<div class="hero-actions">' +
    button("Join the waitlist") +
    routeLink("See how it works " + icon("arrow"), "/how-it-works", "text-link") +
    "</div>" +
    "</div>" +
    '<div class="hero__visual about-hero__visual" data-reveal>' +
    mirrorVisual({ screen: "idle", size: "large" }) +
    phoneVisual({ screen: "home" }) +
    "</div>" +
    "</div>" +
    "</section>" +
    '<section class="section section--tight" data-reveal>' +
    '<div class="principles-grid">' +
    principleCard("people", "Supports connection, not surveillance", "Reflexion is here to spark meaningful conversations, not track or monitor.") +
    principleCard("bell", "Flags changes, not diagnoses", "We highlight patterns that may matter—so families can check in and respond.") +
    principleCard("shield", "Built for privacy and consent", "Personal data stays private and is shared only with your permission.") +
    principleCard("heart", "Designed for real families", "Every detail—from the mirror to the app—was shaped by caregivers like you.") +
    "</div>" +
    "</section>" +
    '<section class="section section--tight" data-reveal>' +
    '<div class="story">' +
    '<div><p class="panel-label">OUR STORY</p><h2>Born from a simple but powerful belief.</h2><p>Reflexion was created by families who know the quiet worry of living with someone from afar.</p><p>We believe that a daily check-in can be a bridge—bringing reassurance, preserving independence, and strengthening the bonds that matter most.</p><p class="story-signoff">With care,<span>The Reflexion Team</span></p></div>' +
    '<div class="story-visual">' +
    mirrorVisual({ screen: "idle", size: "large" }) +
    phoneVisual({ screen: "home" }) +
    "</div>" +
    "</div>" +
    "</section>" +
    '<section class="section section--tight" data-reveal>' +
    '<div class="section-heading">' +
    "<h2>Trust is at the heart of everything we build.</h2>" +
    "</div>" +
    trustStrip([
      { icon: "shield", title: "Privacy first", copy: "Your data is private, secure, and never sold." },
      { icon: "heart", title: "Dignity always", copy: "We design with respect for independence." },
      { icon: "people", title: "Family-centered", copy: "Tools that empower caregivers and loved ones." },
      { icon: "check", title: "Real-life ready", copy: "Simple, warm, and built for everyday use." },
    ]) +
    "</section>" +
    ctaBand("Care. Connected.", "A gentler way to stay close.", "See the mirror", "/the-mirror") +
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

function render() {
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
  if (event.key === "Escape") closeWaitlist();
});

window.addEventListener("popstate", render);

render();
