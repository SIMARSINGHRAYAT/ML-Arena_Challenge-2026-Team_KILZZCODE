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

  // --- High-Fidelity YouTube Audio Bridge ---
  let ytApiReady = false;
  const pendingYTInits = [];

  function ensureYTReady(cb) {
    if (ytApiReady) {
      cb();
    } else {
      pendingYTInits.push(cb);
    }
  }

  // Hook into YouTube IFrame API callback
  const prevOnYouTubeReady = window.onYouTubeIframeAPIReady;
  window.onYouTubeIframeAPIReady = function () {
    if (typeof prevOnYouTubeReady === 'function') {
      try { prevOnYouTubeReady(); } catch {}
    }
    ytApiReady = true;
    for (const fn of pendingYTInits) {
      try { fn(); } catch (err) { console.warn('[bytune-audio] init error:', err); }
    }
    pendingYTInits.length = 0;
  };

  if (window.YT && window.YT.Player) {
    ytApiReady = true;
  }

  function getPlayerContainer() {
    let container = document.getElementById('bytune-yt-player-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'bytune-yt-player-container';
      container.style.cssText = 'position:fixed;left:-9999px;bottom:-9999px;width:200px;height:200px;opacity:0.001;pointer-events:none;z-index:-9999;';
      (document.body || document.documentElement).appendChild(container);
    }
    return container;
  }

  function extractVideoId(url) {
    if (!url || typeof url !== 'string') return null;
    const m1 = url.match(/[?&]id=([a-zA-Z0-9_-]{11})/);
    if (m1) return m1[1];
    const m2 = url.match(/\/([a-zA-Z0-9_-]{11})(?:[?#]|$)/);
    if (m2) return m2[1];
    if (url.length === 11 && /^[a-zA-Z0-9_-]{11}$/.test(url)) return url;
    return null;
  }

  let audioCounter = 0;
  const OrigAudio = window.Audio;

  class AudioBridge extends EventTarget {
    constructor(initialSrc) {
      super();
      this._id = ++audioCounter;
      this._slotId = 'bytune-yt-slot-' + this._id;
      this._src = '';
      this._videoId = null;
      this._paused = true;
      this._ended = false;
      this._currentTime = 0;
      this._duration = 0;
      this._volume = 1;
      this._muted = false;
      this._preload = 'auto';
      this._playbackRate = 1;
      this._playerReady = false;
      this._ytPlayer = null;
      this._pendingVideoId = null;
      this._pendingPlay = false;
      this._pendingSeek = null;
      this._timeUpdateInterval = null;
      this._nativeAudio = new OrigAudio();

      const nativeEvents = ['timeupdate', 'progress', 'waiting', 'canplay', 'playing', 'pause', 'ended', 'error', 'volumechange', 'durationchange'];
      for (const ev of nativeEvents) {
        this._nativeAudio.addEventListener(ev, () => {
          if (!this._videoId) {
            this.dispatchEvent(new Event(ev));
          }
        });
      }

      this._initSlot();
      if (initialSrc) {
        this.src = initialSrc;
      }
    }

    _initSlot() {
      const setup = () => {
        const container = getPlayerContainer();
        const slot = document.createElement('div');
        slot.id = this._slotId;
        container.appendChild(slot);

        ensureYTReady(() => {
          try {
            this._ytPlayer = new window.YT.Player(this._slotId, {
              width: '200',
              height: '200',
              playerVars: {
                autoplay: 0,
                controls: 0,
                disablekb: 1,
                fs: 0,
                modestbranding: 1,
                rel: 0,
                playsinline: 1,
                enablejsapi: 1,
                origin: window.location.origin
              },
              events: {
                onReady: () => {
                  this._playerReady = true;
                  if (this._volume !== 1) this._ytPlayer.setVolume(Math.round(this._volume * 100));
                  if (this._muted) this._ytPlayer.mute();
                  if (this._pendingVideoId) {
                    const vid = this._pendingVideoId;
                    const shouldPlay = this._pendingPlay;
                    const seek = this._pendingSeek || 0;
                    this._pendingVideoId = null;
                    this._pendingPlay = false;
                    this._pendingSeek = null;
                    if (shouldPlay) {
                      this._ytPlayer.loadVideoById(vid, seek);
                    } else {
                      this._ytPlayer.cueVideoById(vid, seek);
                    }
                  }
                },
                onStateChange: (event) => {
                  const state = event.data;
                  if (state === window.YT.PlayerState.PLAYING) {
                    this._paused = false;
                    this._ended = false;
                    this._startTimeUpdates();
                    this.dispatchEvent(new Event('playing'));
                    this.dispatchEvent(new Event('canplay'));
                  } else if (state === window.YT.PlayerState.PAUSED) {
                    this._paused = true;
                    this._stopTimeUpdates();
                    this.dispatchEvent(new Event('pause'));
                  } else if (state === window.YT.PlayerState.BUFFERING) {
                    this.dispatchEvent(new Event('waiting'));
                  } else if (state === window.YT.PlayerState.ENDED) {
                    this._paused = true;
                    this._ended = true;
                    this._stopTimeUpdates();
                    this.dispatchEvent(new Event('ended'));
                  } else if (state === window.YT.PlayerState.CUED) {
                    this.dispatchEvent(new Event('canplay'));
                  }
                },
                onError: (event) => {
                  console.warn('[bytune-audio] YouTube playback error:', event.data);
                  this.dispatchEvent(new Event('error'));
                }
              }
            });
          } catch (err) {
            console.warn('[bytune-audio] YT Player creation failed:', err);
          }
        });
      };

      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setup, { once: true });
      } else {
        setup();
      }
    }

    _startTimeUpdates() {
      this._stopTimeUpdates();
      this._timeUpdateInterval = setInterval(() => {
        if (this._playerReady && this._ytPlayer && typeof this._ytPlayer.getCurrentTime === 'function') {
          try {
            const cur = this._ytPlayer.getCurrentTime();
            if (typeof cur === 'number' && !isNaN(cur)) {
              this._currentTime = cur;
            }
          } catch {}
        }
        this.dispatchEvent(new Event('timeupdate'));
        this.dispatchEvent(new Event('progress'));
      }, 200);
    }

    _stopTimeUpdates() {
      if (this._timeUpdateInterval) {
        clearInterval(this._timeUpdateInterval);
        this._timeUpdateInterval = null;
      }
    }

    get src() {
      return this._src;
    }

    set src(url) {
      this._src = String(url || '');
      const vid = extractVideoId(this._src);
      if (vid) {
        this._videoId = vid;
        this._ended = false;
        this._currentTime = 0;
        if (this._playerReady && this._ytPlayer && typeof this._ytPlayer.loadVideoById === 'function') {
          if (!this._paused || this._pendingPlay) {
            this._ytPlayer.loadVideoById(vid, 0);
            this._pendingPlay = false;
          } else {
            this._ytPlayer.cueVideoById(vid, 0);
          }
        } else {
          this._pendingVideoId = vid;
        }
      } else {
        this._videoId = null;
        this._nativeAudio.src = this._src;
      }
    }

    play() {
      if (this._videoId) {
        this._paused = false;
        this._ended = false;
        if (this._playerReady && this._ytPlayer && typeof this._ytPlayer.playVideo === 'function') {
          const curVid = (this._ytPlayer.getVideoData && this._ytPlayer.getVideoData().video_id) || null;
          if (curVid !== this._videoId) {
            this._ytPlayer.loadVideoById(this._videoId, this._currentTime || 0);
          } else {
            this._ytPlayer.playVideo();
          }
          this._startTimeUpdates();
        } else {
          this._pendingPlay = true;
        }
        this.dispatchEvent(new Event('play'));
        return Promise.resolve();
      }
      return this._nativeAudio.play();
    }

    pause() {
      if (this._videoId) {
        this._paused = true;
        this._pendingPlay = false;
        if (this._playerReady && this._ytPlayer && typeof this._ytPlayer.pauseVideo === 'function') {
          this._ytPlayer.pauseVideo();
        }
        this._stopTimeUpdates();
        this.dispatchEvent(new Event('pause'));
        return;
      }
      this._nativeAudio.pause();
    }

    load() {
      if (!this._videoId) {
        this._nativeAudio.load();
      }
    }

    get currentTime() {
      if (this._videoId) {
        if (this._playerReady && this._ytPlayer && typeof this._ytPlayer.getCurrentTime === 'function') {
          try {
            const t = this._ytPlayer.getCurrentTime();
            if (typeof t === 'number' && !isNaN(t)) {
              this._currentTime = t;
              return t;
            }
          } catch {}
        }
        return this._currentTime || 0;
      }
      return this._nativeAudio.currentTime;
    }

    set currentTime(val) {
      const t = Math.max(0, Number(val) || 0);
      this._currentTime = t;
      if (this._videoId) {
        if (this._playerReady && this._ytPlayer && typeof this._ytPlayer.seekTo === 'function') {
          this._ytPlayer.seekTo(t, true);
        } else {
          this._pendingSeek = t;
        }
        this.dispatchEvent(new Event('timeupdate'));
        return;
      }
      this._nativeAudio.currentTime = t;
    }

    get duration() {
      if (this._videoId) {
        if (this._playerReady && this._ytPlayer && typeof this._ytPlayer.getDuration === 'function') {
          try {
            const d = this._ytPlayer.getDuration();
            if (typeof d === 'number' && d > 0) {
              this._duration = d;
              return d;
            }
          } catch {}
        }
        return this._duration || 0;
      }
      return this._nativeAudio.duration;
    }

    get volume() {
      return this._volume;
    }

    set volume(val) {
      const v = Math.max(0, Math.min(1, Number(val) || 0));
      this._volume = v;
      if (this._videoId && this._playerReady && this._ytPlayer && typeof this._ytPlayer.setVolume === 'function') {
        this._ytPlayer.setVolume(Math.round(v * 100));
      }
      this._nativeAudio.volume = v;
      this.dispatchEvent(new Event('volumechange'));
    }

    get muted() {
      return this._muted;
    }

    set muted(val) {
      const m = Boolean(val);
      this._muted = m;
      if (this._videoId && this._playerReady && this._ytPlayer) {
        if (m && typeof this._ytPlayer.mute === 'function') this._ytPlayer.mute();
        else if (!m && typeof this._ytPlayer.unMute === 'function') this._ytPlayer.unMute();
      }
      this._nativeAudio.muted = m;
      this.dispatchEvent(new Event('volumechange'));
    }

    get paused() {
      return this._videoId ? this._paused : this._nativeAudio.paused;
    }

    get ended() {
      return this._videoId ? this._ended : this._nativeAudio.ended;
    }

    get buffered() {
      const d = this.duration || 0;
      return {
        length: 1,
        start: () => 0,
        end: () => d,
      };
    }

    get preload() {
      return this._preload;
    }

    set preload(val) {
      this._preload = val;
      this._nativeAudio.preload = val;
    }

    get playbackRate() {
      return this._playbackRate;
    }

    set playbackRate(val) {
      this._playbackRate = val;
      if (this._videoId && this._playerReady && this._ytPlayer && typeof this._ytPlayer.setPlaybackRate === 'function') {
        this._ytPlayer.setPlaybackRate(val);
      }
      this._nativeAudio.playbackRate = val;
    }
  }

  // Hook prototype for on<event> properties
  const eventNames = ['timeupdate', 'progress', 'waiting', 'canplay', 'playing', 'pause', 'ended', 'error', 'volumechange', 'durationchange', 'seeking', 'seeked'];
  for (const name of eventNames) {
    Object.defineProperty(AudioBridge.prototype, 'on' + name, {
      get() { return this['_on' + name] || null; },
      set(fn) {
        if (this['_on' + name]) this.removeEventListener(name, this['_on' + name]);
        this['_on' + name] = typeof fn === 'function' ? fn : null;
        if (this['_on' + name]) this.addEventListener(name, this['_on' + name]);
      }
    });
  }

  // Replace standard Audio constructor
  window.Audio = function (src) {
    return new AudioBridge(src);
  };
  window.Audio.prototype = AudioBridge.prototype;

  const origCreateElement = document.createElement.bind(document);
  document.createElement = function (tagName, options) {
    if (typeof tagName === 'string' && tagName.toLowerCase() === 'audio') {
      return new AudioBridge();
    }
    return origCreateElement(tagName, options);
  };

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
        let plays = 0;
        const trackMap = new Map();
        const artistMap = new Map();
        const albumMap = new Map();

        for (const entry of history) {
          totalMs += entry.deltaMs || 0;
          plays += 1;
          const tr = entry.track;
          if (tr) {
            const trackId = tr.id || tr.title;
            const curTr = trackMap.get(trackId) || { ...tr, ms: 0, plays: 0 };
            curTr.ms += entry.deltaMs || 0;
            curTr.plays += 1;
            trackMap.set(trackId, curTr);

            if (tr.artist && tr.artist !== 'Unknown artist') {
              const curArt = artistMap.get(tr.artist) || { name: tr.artist, ms: 0, plays: 0 };
              curArt.ms += entry.deltaMs || 0;
              curArt.plays += 1;
              artistMap.set(tr.artist, curArt);
            }

            if (tr.album) {
              const curAlb = albumMap.get(tr.album) || { name: tr.album, artist: tr.artist, ms: 0, plays: 0 };
              curAlb.ms += entry.deltaMs || 0;
              curAlb.plays += 1;
              albumMap.set(tr.album, curAlb);
            }
          }
        }

        return {
          totalMs,
          plays,
          tracks: Array.from(trackMap.values()).sort((a, b) => b.ms - a.ms).slice(0, 50),
          artists: Array.from(artistMap.values()).sort((a, b) => b.ms - a.ms).slice(0, 50),
          albums: Array.from(albumMap.values()).sort((a, b) => b.ms - a.ms).slice(0, 50),
          hours: new Array(24).fill(0),
          days: [],
          period,
          months: [],
        };
      } catch {
        return {
          totalMs: 0,
          plays: 0,
          tracks: [],
          artists: [],
          albums: [],
          hours: new Array(24).fill(0),
          days: [],
          period,
          months: [],
        };
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

