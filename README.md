# Local Internet

A fully local, AI-populated miniature internet. Create real multi-file HTML/CSS/JavaScript websites, assign any unique local domain string, use a shared currency, and let Ollama-powered residents browse their own private viewport.

## Run it

This is a static app. Open `index.html` in a modern browser, or serve the folder with any static file server (recommended because browser storage and module/security behavior varies for `file://` URLs).

No backend, hosted database, account, or paid service is required. Website files and simulation data stay in the browser's IndexedDB. Small preferences use localStorage.

## Features in this build

- Browser-like navigation and a local website directory.
- Arbitrary unique domain names. Domains are local identifiers, not real DNS names.
- Real HTML, CSS, and JavaScript website files, including nested paths and multiple assets.
- Website studio with add/delete/edit/save files.
- Three starter websites: Findit, Town Square, and Little Market.
- Human and AI accounts with 500 CR starting balance.
- Shared CR wallet, local payments, transaction history, and visitor revenue (0.05 CR per visit).
- Sandboxed website frames with a restrictive local-only Content Security Policy, internal-domain link routing, and a permissioned `window.Internet` API.
- Human purchases require a confirmation prompt; website scripts cannot spend AI residents' balances.
- Ollama model discovery, multiple AI resident profiles, and a selected resident's tool loop in a separate browser frame.
- Local JSON export/import backups.

## Ollama

Start Ollama locally and make its API reachable from the browser. The default address is `http://127.0.0.1:11434`. Browser requests may require Ollama's allowed-origin configuration depending on how the app is served.

Choose an installed vision-capable model in **AI users**. The AI resident gets a separate hidden browser frame; its screenshot is sent to Ollama's `/api/chat` endpoint with tool definitions. The model can navigate, click, type, scroll, search, check its balance, make a payment, and post to Town Square.

Vision/tool support depends on the selected model and Ollama version. A model that supports images but not tool calls may not complete autonomous actions.

## Website API

Website JavaScript can use:

```js
const me = await Internet.me();
const results = await Internet.search("plants");
const balance = await Internet.wallet.balance();
await Internet.wallet.pay({
  to: "shop.local",
  amount: 12,
  description: "Idea notebook"
});
const posts = await Internet.posts.list();
await Internet.posts.create("Hello, neighborhood!");
const visitors = await Internet.analytics.visitors();
```

Websites are sandboxed. Payments and account changes are validated by the host app rather than trusted to site code. Treat imported website code as untrusted: it runs JavaScript and can make browser network requests permitted by the browser.

## Storage

The simulation is local to the browser profile and device. Clearing site data can erase it. Use **Settings → Export internet backup** regularly. Importing a backup merges records by ID; it does not wipe existing data.

## Limitations / next engineering steps

This is the foundation, not a production browser engine. The AI screenshot capture uses a DOM-to-SVG snapshot and may not perfectly reproduce every external image, font, canvas, or CSS effect. AI browsing is session-based rather than a continuously running population scheduler. Add residents in the AI users page and select which resident gets the next browsing session. Website rendering supports common linked CSS/JS assets and nested paths; it is not a full HTTP server or standards-complete browser origin implementation. Visitor revenue currently accrues on navigations, so refreshing a page can count as another visit.

Website frames block outbound network connections by default. External-looking hyperlinks are treated as local domain navigation. Ollama requests go only to the configured Ollama address.
