import { ScrollViewStyleReset } from 'expo-router/html';
import type { ReactNode } from 'react';

export default function Root({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta httpEquiv="X-UA-Compatible" content="IE=edge" />
        <meta
          name="viewport"
          content="width=device-width, initial-scale=1, shrink-to-fit=no, viewport-fit=cover"
        />
        <link rel="manifest" href="/manifest.json" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        <meta name="apple-mobile-web-app-title" content="Reflexion" />
        <link rel="apple-touch-icon" href="/icons/icon-192.png" />
        <meta name="theme-color" content="#E6F4FE" />
        <ScrollViewStyleReset />
        <script
          dangerouslySetInnerHTML={{
            __html: `
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(function() {});
}
`,
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
