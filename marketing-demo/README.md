# Reflexion marketing demo

This is a local-only, dependency-free marketing-site demo. It rebuilds the supplied visual references with editable HTML, CSS, and JavaScript components rather than placing the reference screenshots on the page.

## Run it

    cd /Users/chloetan/Documents/Reflexion/marketing-demo
    npm run dev

Open http://127.0.0.1:4173.

No backend or waitlist submission is connected. The waitlist form is a local UX placeholder.

## Main files to edit

- Wording and page content: app.js
- Colours, type, spacing, responsive layout, and motion: styles.css
- Navigation and route labels: app.js (navItems and routes)
- Product/app images: assets/ and the mirrorScreens / phoneScreens maps in app.js
- Page sections and reusable components: app.js
- Local server port and static-file behaviour: server.mjs

The product UI images are sourced from the approved docs/product-source-of-truth/ PNG references. They are shown inside editable device shells so the marketing composition remains easy to change.
