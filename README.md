# 🎵 ByTune Web Player (Vercel-Ready)

ByTune has been converted from a Windows desktop executable (`ByTune-Setup.exe`) into a web application ready for deployment on **Vercel** or self-hosting on any Node.js environment.

---

## 🚀 Live Local Quickstart

To run the web app on your machine:

```bash
# 1. Install dependencies (if not already installed)
npm install

# 2. Start the local server
npm start
```

Visit **`http://localhost:3000`** in any web browser.

---

## ☁️ Deploying to Vercel

### Method 1: Deploy with Vercel CLI

```bash
# 1. Install Vercel CLI (if not installed)
npm i -g vercel

# 2. Deploy to preview
vercel

# 3. Deploy to production
vercel --prod
```

### Method 2: Deploy with GitHub & Vercel Dashboard

1. Initialize git and push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Convert ByTune to Vercel web app"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```
2. Go to [vercel.com/new](https://vercel.com/new).
3. Import your repository.
4. Framework Preset: **Other**.
5. Click **Deploy**. Vercel will automatically build the static assets from `public/` and deploy the serverless API from `api/index.js`.

---

## 🛠️ Architecture & Conversion Overview

### 1. Frontend (`public/`)
- Extracted and decoupled from the Electron shell (`app.asar`).
- Fully featured modern music player UI with responsive layout, animations, queue management, volume controls, search, and artist/album views.
- **Web Bridge (`public/bytune-web-bridge.js`)**: Injects the `window.bytune` API expected by the client, routing data persistence to browser `localStorage`, window controls to standard HTML5 Fullscreen/PiP APIs, and music operations to `/api/music/*`.

### 2. Backend & Serverless API (`api/` & `server/`)
- Powered by Innertube (YouTube Music client) with BotGuard PO token minting and client rotation.
- **Vercel Serverless Route (`api/index.js`)**:
  - `GET /api/music/home`: Curated home feeds and "Quick picks".
  - `GET /api/music/charts`: Global & regional trending charts.
  - `GET /api/music/moods`: Mood & genre playlists.
  - `GET /api/music/search`: Search songs, albums, artists, and playlists.
  - `GET /api/music/suggestions`: Real-time search query auto-completion.
  - `GET /api/music/album` & `artist` & `playlist`: Detailed browse views.
  - `POST /api/music/lyrics`: Synchronized lyrics with word-by-word timestamps.
  - `GET /api/music/stream`: Low-latency audio streaming proxy with HTTP `206 Partial Content` (Range request) support for instant seeking in web browsers.

### 3. Vercel Configuration (`vercel.json`)
- Clean routing config mapping `/api/*` to the serverless backend function while falling back to `index.html` for single-page client-side routing.

---

## 📦 Project Structure

```
├── api/
│   └── index.js                 # Vercel Serverless API handler
├── public/
│   ├── index.html               # Main Web App HTML entrypoint
│   ├── bytune-web-bridge.js     # Window.bytune client-side bridge
│   └── assets/                  # Compiled React & CSS bundles
├── server/
│   ├── engine.js                # ByTune Music Engine
│   ├── bytune-engine-bundle.js  # Bundled InnerTube & media resolver
│   └── dev-server.js            # Standalone dev/production server
├── package.json                 # Node dependencies & npm scripts
├── vercel.json                  # Vercel deployment & routing config
└── README.md
```

