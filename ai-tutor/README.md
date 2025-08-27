# AI Tutor – Simple Static Site

A minimal, responsive landing page with a built‑in chat demo that simulates an AI tutor locally in your browser.

## Run locally

- Using Python 3:
```bash
cd ai-tutor
python3 -m http.server 8000
```
- Then open `http://localhost:8000` in your browser.

No build step or dependencies required.

## Project structure

```
ai-tutor/
├─ index.html
├─ styles.css
├─ script.js
└─ assets/
   └─ logo.svg
```

## Customize

- Update branding in `index.html` and `assets/logo.svg`.
- Tweak colors and spacing in `styles.css`.
- Replace the simulated reply logic in `script.js` with your API calls when ready.

## Deploy

Host the folder on any static host (GitHub Pages, Netlify, Vercel, S3/CloudFront). Just upload the files as-is.