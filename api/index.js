const engine = require('../server/engine');

function parseBody(req) {
  return new Promise((resolve) => {
    if (req.body && typeof req.body === 'object') {
      return resolve(req.body);
    }
    let data = '';
    req.on('data', (chunk) => {
      data += chunk;
    });
    req.on('end', () => {
      try {
        resolve(data ? JSON.parse(data) : {});
      } catch {
        resolve({});
      }
    });
  });
}

module.exports = async function handler(req, res) {
  // Global CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Range, Authorization');

  if (req.method === 'OPTIONS') {
    res.statusCode = 200;
    res.end();
    return;
  }

  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  const pathname = url.pathname;

  try {
    // Audio streaming handler (handles HTTP Range and proxying audio)
    if (pathname === '/api/music/stream' || pathname.endsWith('/stream')) {
      req.query = req.query || Object.fromEntries(url.searchParams.entries());
      return await engine.handleStream(req, res);
    }

    // JSON response helper
    const sendJson = (data, status = 200) => {
      res.statusCode = status;
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify(data));
    };

    if (pathname === '/api/health') {
      return sendJson({ status: 'ok', name: 'ByTune Web API', version: '1.0.1' });
    }

    if (pathname === '/api/debug/stream') {
      const id = url.searchParams.get('id') || 'BSTsnWoslP4';
      const yt = await require('../server/bytune-engine-bundle').getYT?.() || null;
      // In bytune-engine-bundle, let's test getBasicInfo on all clients
      const results = {};
      const clients = ["ANDROID", "IOS", "VISIONOS", "WEB", "YTMUSIC", "TV"];
      for (const c of clients) {
        try {
          const info = await yt.getBasicInfo(id, { client: c });
          const sd = info?.streaming_data;
          results[c] = {
            status: info?.playability_status?.status,
            reason: info?.playability_status?.reason,
            formats: (sd?.adaptive_formats || []).length,
            hasAudio: (sd?.adaptive_formats || []).some(f => f.has_audio),
          };
        } catch (e) {
          results[c] = { error: e.message };
        }
      }
      return sendJson(results);
    }

    if (pathname === '/api/music/home') {
      const data = await engine.getHome();
      return sendJson(data);
    }

    if (pathname === '/api/music/charts') {
      const data = await engine.getCharts();
      return sendJson(data);
    }

    if (pathname === '/api/music/moods') {
      const data = await engine.getMoods();
      return sendJson(data);
    }

    if (pathname === '/api/music/search') {
      const query = url.searchParams.get('query') || '';
      const light = url.searchParams.get('light') === 'true';
      const data = await engine.search(query, light);
      return sendJson(data || { songs: [], artists: [], albums: [], playlists: [] });
    }

    if (pathname === '/api/music/searchPage') {
      const query = url.searchParams.get('query') || '';
      const filter = url.searchParams.get('filter') || void 0;
      const data = await engine.searchPage(query, filter);
      return sendJson(data || { rows: [] });
    }

    if (pathname === '/api/music/searchMore') {
      const token = url.searchParams.get('token') || '';
      const filter = url.searchParams.get('filter') || void 0;
      const data = await engine.searchMore(token, filter);
      return sendJson(data || { rows: [] });
    }

    if (pathname === '/api/music/suggestions') {
      const query = url.searchParams.get('query') || '';
      const data = await engine.suggestions(query);
      return sendJson(data || { items: [] });
    }

    if (pathname === '/api/music/artist') {
      const id = url.searchParams.get('id') || '';
      const data = await engine.getArtist(id);
      return sendJson(data);
    }

    if (pathname === '/api/music/album') {
      const id = url.searchParams.get('id') || '';
      const data = await engine.getAlbum(id);
      return sendJson(data);
    }

    if (pathname === '/api/music/playlist') {
      const id = url.searchParams.get('id') || '';
      const data = await engine.getPlaylist(id);
      return sendJson(data);
    }

    if (pathname === '/api/music/playlistPage') {
      const id = url.searchParams.get('id') || '';
      const data = await engine.getPlaylistPage(id);
      return sendJson(data);
    }

    if (pathname === '/api/music/market') {
      const data = await engine.marketInfo();
      return sendJson(data);
    }

    if (pathname === '/api/music/lyrics') {
      let meta = {};
      if (req.method === 'POST') {
        meta = await parseBody(req);
      } else {
        meta = {
          id: url.searchParams.get('id'),
          title: url.searchParams.get('title'),
          artist: url.searchParams.get('artist'),
          album: url.searchParams.get('album'),
          duration: Number(url.searchParams.get('duration') || 0),
        };
      }
      const data = await engine.getLyrics(meta);
      return sendJson(data);
    }

    if (pathname === '/api/music/canvas') {
      let q = {};
      if (req.method === 'POST') {
        q = await parseBody(req);
      } else {
        q = {
          title: url.searchParams.get('title'),
          artist: url.searchParams.get('artist'),
          album: url.searchParams.get('album'),
        };
      }
      const data = await engine.getCanvas(q);
      return sendJson(data);
    }

    if (pathname === '/api/music/enrich') {
      let ids = [];
      if (req.method === 'POST') {
        const body = await parseBody(req);
        ids = body.ids || [];
      } else {
        const rawIds = url.searchParams.get('ids') || '';
        ids = rawIds.split(',').filter(Boolean);
      }
      const data = await engine.enrichDurations(ids);
      return sendJson(data);
    }

    // Default 404 for unknown API route
    return sendJson({ error: 'Endpoint not found', path: pathname }, 404);
  } catch (err) {
    console.error(`[api-error] ${pathname}:`, err);
    res.statusCode = 500;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ error: err.message || 'Internal Server Error' }));
  }
};

