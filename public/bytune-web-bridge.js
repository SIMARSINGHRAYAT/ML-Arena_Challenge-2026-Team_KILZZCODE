/**
 * ByTune Web Bridge
 * Provides the `window.bytune` API surface in web browsers,
 * connecting the ByTune React client to standard Web APIs and the serverless backend.
 */
(function () {
  'use strict';

  async function apiCall(endpoint, params = null, method = 'GET') {
    let url = endpoint;
    const options = {
      method,
      headers: {
        'Accept': 'application/json',
      },
    };

    if (method === 'GET' && params) {
      const search = new URLSearchParams();
      for (const [key, val] of Object.entries(params)) {
        if (val !== undefined && val !== null) {
          search.append(key, String(val));
        }
      }
      const qs = search.toString();
      if (qs) url += (url.includes('?') ? '&' : '?') + qs;
    } else if (method === 'POST' && params) {
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(params);
    }

    try {
      const res = await fetch(url, options);
      if (!res.ok) {
        throw new Error(`API error ${res.status}: ${res.statusText}`);
      }
      return await res.json();
    } catch (err) {
      console.warn(`[bytune-web] API call failed (${endpoint}):`, err);
      throw err;
    }
  }

  // Event dispatchers for auth & sync
  const authListeners = new Set();
  const syncListeners = new Set();
  const fsListeners = new Set();

  function notifyAuth() {
    for (const cb of authListeners) {
      try { cb(); } catch (e) { console.error(e); }
    }
  }

  document.addEventListener('fullscreenchange', () => {
    const isFs = !!document.fullscreenElement;
    for (const cb of fsListeners) {
      try { cb(isFs); } catch (e) { console.error(e); }
    }
  });

  window.bytune = {
    platform: 'web',
    versions: {
      app: '1.0.1',
      electron: '',
      chrome: '',
      node: '',
    },

    // Music APIs (proxied to serverless backend)
    search: (query, light = false) => apiCall('/api/music/search', { query, light }),
    searchPage: (query, filter) => apiCall('/api/music/searchPage', { query, filter }),
    searchMore: (token, filter) => apiCall('/api/music/searchMore', { token, filter }),
    suggestions: (query) => apiCall('/api/music/suggestions', { query }),
    getAlbum: (id) => apiCall('/api/music/album', { id }),
    getArtist: (id) => apiCall('/api/music/artist', { id }),
    getPlaylist: (id) => apiCall('/api/music/playlist', { id }),
    getPlaylistPage: (id) => apiCall('/api/music/playlistPage', { id }),
    getHome: () => apiCall('/api/music/home'),
    getCharts: () => apiCall('/api/music/charts'),
    getMoods: () => apiCall('/api/music/moods'),
    marketInfo: () => apiCall('/api/music/market').catch(() => ({ country: 'US', source: 'default' })),
    getLyrics: (track) => apiCall('/api/music/lyrics', track, 'POST'),
    enrichDurations: (ids) => apiCall('/api/music/enrich', { ids }, 'POST'),
    getCanvas: (q) => apiCall('/api/music/canvas', q, 'POST').catch(() => null),

    getStreamUrl: (id, forceRotate = false, quality) => {
      let url = `/api/music/stream?id=${encodeURIComponent(id)}`;
      if (forceRotate) url += '&rotate=1';
      if (quality) url += `&quality=${encodeURIComponent(quality)}`;
      return Promise.resolve(url);
    },

    // Persistence via localStorage
    readData: async (name) => {
      try {
        const item = localStorage.getItem('bytune_data_' + name);
        return item ? JSON.parse(item) : null;
      } catch {
        return null;
      }
    },
    writeData: async (name, data) => {
      try {
        if (data === null || data === undefined) {
          localStorage.removeItem('bytune_data_' + name);
        } else {
          localStorage.setItem('bytune_data_' + name, JSON.stringify(data));
        }
        return true;
      } catch {
        return false;
      }
    },
    resetData: async () => {
      try {
        const keys = Object.keys(localStorage).filter((k) => k.startsWith('bytune_'));
        for (const k of keys) localStorage.removeItem(k);
        return true;
      } catch {
        return false;
      }
    },

    // Stats & Listening history
    playbackEvent: (event) => {
      try {
        if (event?.type === 'progress' && event?.track && (event.deltaMs ?? 0) > 0) {
          const stats = JSON.parse(localStorage.getItem('bytune_history') || '[]');
          stats.push({
            id: event.track.id,
            track: event.track,
            deltaMs: event.deltaMs,
            timestamp: Date.now(),
          });
          if (stats.length > 500) stats.splice(0, stats.length - 500);
          localStorage.setItem('bytune_history', JSON.stringify(stats));
        }
      } catch {}
    },

    statsSummary: async (period = 'month') => {
      try {
        const history = JSON.parse(localStorage.getItem('bytune_history') || '[]');
        let totalMs = 0;
        const songCounts = new Map();

        for (const entry of history) {
          totalMs += entry.deltaMs || 0;
          if (entry.track?.id) {
            const current = songCounts.get(entry.track.id) || { count: 0, track: entry.track };
            current.count++;
            songCounts.set(entry.track.id, current);
          }
        }

        const topTracks = Array.from(songCounts.values())
          .sort((a, b) => b.count - a.count)
          .slice(0, 10)
          .map((x) => x.track);

        return {
          totalSeconds: Math.round(totalMs / 1000),
          topTracks,
          period,
        };
      } catch {
        return { totalSeconds: 0, topTracks: [], period };
      }
    },

    // Backup & Restore
    backupExport: async () => {
      try {
        const backup = {};
        for (let i = 0; i < localStorage.length; i++) {
          const key = localStorage.key(i);
          if (key && key.startsWith('bytune_')) {
            backup[key] = localStorage.getItem(key);
          }
        }
        const blob = new Blob([JSON.stringify(backup, null, 2)], { type: 'application/json' });
        const blobUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = blobUrl;
        a.download = `bytune-backup-${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(blobUrl);
        return { path: 'backup.json' };
      } catch (err) {
        return null;
      }
    },

    backupImport: async () => {
      return new Promise((resolve) => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        input.onchange = async () => {
          const file = input.files?.[0];
          if (!file) return resolve(null);
          try {
            const text = await file.text();
            const data = JSON.parse(text);
            for (const [key, val] of Object.entries(data)) {
              if (typeof val === 'string') {
                localStorage.setItem(key, val);
              }
            }
            resolve({ restored: true, message: 'Settings restored' });
            window.location.reload();
          } catch (e) {
            resolve(null);
          }
        };
        input.click();
      });
    },

    // Auth & Accounts (runs in guest mode in browser, persists username locally)
    authGetSession: async () => {
      try {
        const profile = JSON.parse(localStorage.getItem('bytune_profile') || 'null');
        return {
          mode: 'guest',
          userId: profile?.id || 'guest',
          username: profile?.username || 'Guest',
          isNewAccount: false,
        };
      } catch {
        return { mode: 'guest', userId: 'guest', username: 'Guest', isNewAccount: false };
      }
    },
    authContinueGuest: async () => ({ mode: 'guest', userId: 'guest' }),
    authClearGuestProfile: async () => {
      localStorage.removeItem('bytune_profile');
      notifyAuth();
    },
    authUsernameAvailable: async () => true,
    authSetUsername: async (username) => {
      const profile = JSON.parse(localStorage.getItem('bytune_profile') || '{}');
      profile.username = username;
      if (!profile.id) profile.id = 'guest_' + Math.random().toString(36).slice(2, 8);
      localStorage.setItem('bytune_profile', JSON.stringify(profile));
      notifyAuth();
    },
    authSkipUsername: async () => {},
    authSignOut: async () => {
      localStorage.removeItem('bytune_profile');
      notifyAuth();
    },
    authSignIn: async () => {
      throw new Error('Cloud sync accounts are supported in desktop app. You are currently in Guest web mode.');
    },
    authSignUp: async () => {
      throw new Error('Cloud sync accounts are supported in desktop app. You are currently in Guest web mode.');
    },
    authSignInGoogle: async () => {
      throw new Error('Google Sign-In is supported in desktop app.');
    },
    onAuthChanged: (cb) => {
      authListeners.add(cb);
      return () => authListeners.delete(cb);
    },
    syncStatus: async () => ({ lastSyncAt: null, isSyncing: false }),
    syncNow: async () => {},
    syncBackupNow: async () => {},
    syncRestoreNow: async () => {},
    onSyncRestored: (cb) => {
      syncListeners.add(cb);
      return () => syncListeners.delete(cb);
    },

    // Window controls
    windowMinimize: () => {},
    windowMaximizeToggle: () => {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    },
    windowClose: () => {
      window.close();
    },
    isMaximized: async () => !!document.fullscreenElement,
    setOverlayChrome: async () => 'unsupported',
    setFullScreen: async (fs) => {
      if (fs && !document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else if (!fs && document.fullscreenElement) {
        document.exitFullscreen().catch(() => {});
      }
      return true;
    },
    onMaximizeChange: (cb) => () => {},
    onFullScreenChange: (cb) => {
      fsListeners.add(cb);
      return () => fsListeners.delete(cb);
    },

    // PiP controls
    pipToggle: async () => {
      const video = document.querySelector('video');
      if (video && document.pictureInPictureEnabled) {
        if (document.pictureInPictureElement) {
          await document.exitPictureInPicture().catch(() => {});
        } else {
          await video.requestPictureInPicture().catch(() => {});
        }
      }
    },
    pipIsOpen: async () => !!document.pictureInPictureElement,
    pipClose: () => {},
    pipMinimize: () => {},
    pipShowMain: () => window.focus(),
    pipSendCommand: () => {},
    pipSendState: () => {},
    onPipState: () => () => {},
    onPipCommand: () => () => {},
    onPipOpenChange: (cb) => () => {},
    onPipNeedState: () => () => {},
    onPipCursorInside: () => () => {},

    // Menu
    onMenuCommand: () => () => {},
    menuPushState: () => {},

    // Downloads
    defaultDownloadDir: async () => 'Downloads',
    pickFolder: async () => null,
    revealPath: () => {},
    openPath: async () => 'Not allowed',
    startDownload: async (track, dir, quality) => {
      try {
        const streamUrl = `/api/music/stream?id=${encodeURIComponent(track.id)}${quality ? '&quality=' + quality : ''}`;
        const a = document.createElement('a');
        a.href = streamUrl;
        a.download = `${track.artist || 'Artist'} - ${track.title || 'Track'}.mp3`;
        a.target = '_blank';
        document.body.appendChild(a);
        a.click();
        a.remove();
        return { ok: true };
      } catch (err) {
        return { ok: false, error: err.message };
      }
    },
    cancelDownload: async () => {},
    downloadQueue: async () => [],
    onDownloadEvent: () => () => {},
    onDownloadProgress: () => () => {},

    // Local Library (desktop feature, returns empty defaults for web)
    localLibrary: async () => ({ folders: [], tracks: [] }),
    scanLibrary: async () => ({ added: [], lib: { folders: [], tracks: [] } }),
    setLibraryFolders: async () => ({ folders: [] }),
    addLibraryFolders: async () => ({ added: [], lib: { folders: [], tracks: [] } }),
    renameLibraryFolder: async () => {},
    removeLibraryFolder: async () => {},
    libraryFolderStatuses: async () => [],
    localTrackUrl: async () => null,
    localArtBytes: async () => null,
  };

  console.log('[bytune-web] Web bridge initialized successfully.');
})();

