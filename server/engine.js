const http = require('http');
const Module = require('module');
const path = require('path');
const os = require('os');
const fs = require('fs');

const origRequire = Module.prototype.require;
const ipcHandlers = {};
let streamHandler = null;

// Intercept http.createServer to capture the internal stream proxy handler
const origCreateServer = http.createServer;
http.createServer = function(fn) {
  if (typeof fn === 'function') {
    streamHandler = fn;
  }
  return origCreateServer.apply(this, arguments);
};

// Shimming electron APIs cleanly
const tempUserData = path.join(os.tmpdir(), 'bytune-web-userdata');
try {
  fs.mkdirSync(tempUserData, { recursive: true });
} catch {}

Module.prototype.require = function(modPath) {
  if (modPath === 'electron') {
    return {
      app: {
        getLocaleCountryCode: () => 'US',
        getLocale: () => 'en',
        getVersion: () => '1.0.1',
        getAppPath: () => __dirname,
        isPackaged: true,
        getPath: (name) => {
          if (name === 'userData') return tempUserData;
          return path.join(os.tmpdir(), name || 'bytune');
        },
        whenReady: () => Promise.resolve(),
        on: () => {},
        quit: () => {},
        exit: () => {},
        requestSingleInstanceLock: () => true,
        commandLine: { appendSwitch: () => {} },
        setAsDefaultProtocolClient: () => {},
        setAboutPanelOptions: () => {},
      },
      ipcMain: {
        handle: (channel, fn) => {
          ipcHandlers[channel] = fn;
        },
        on: (channel, fn) => {
          ipcHandlers[channel] = fn;
        },
      },
      BrowserWindow: class {
        constructor() {
          this.webContents = {
            mainFrame: { url: 'file:///dist/index.html' },
            send: () => {},
            session: {
              webRequest: {
                onBeforeSendHeaders: () => {},
                onCompleted: () => {},
                onHeadersReceived: () => {},
              },
            },
            on: () => {},
            setWindowOpenHandler: () => {},
            focus: () => {},
          };
        }
        loadURL() {}
        loadFile() {}
        on() {}
        once() {}
        isMaximized() { return false; }
        show() {}
        focus() {}
        isDestroyed() { return false; }
      },
      dialog: { showOpenDialog: () => Promise.resolve({ canceled: true, filePaths: [] }) },
      Menu: { setApplicationMenu: () => {}, buildFromTemplate: () => ({}) },
      nativeImage: { createEmpty: () => ({}) },
      protocol: {
        registerFileProtocol: () => {},
        registerStreamProtocol: () => {},
        registerSchemesAsPrivileged: () => {},
        handle: () => {},
      },
      safeStorage: { isEncryptionAvailable: () => false },
      screen: { getPrimaryDisplay: () => ({ workArea: { width: 1920, height: 1080 } }) },
      session: {
        defaultSession: {
          webRequest: {
            onBeforeSendHeaders: () => {},
            onCompleted: () => {},
            onHeadersReceived: () => {},
          },
          setPermissionRequestHandler: () => {},
          setPermissionCheckHandler: () => true,
        },
      },
      shell: { openExternal: () => Promise.resolve() },
    };
  }
  return origRequire.apply(this, arguments);
};

// Require the bundled ByTune backend
let initReady = false;
let initPromise = null;

function ensureReady() {
  if (!initPromise) {
    initPromise = new Promise((resolve) => {
      try {
        require('./bytune-engine-bundle.js');
        setImmediate(() => {
          initReady = true;
          resolve();
        });
      } catch (err) {
        console.error('[bytune-engine] bundle load error:', err);
        resolve();
      }
    });
  }
  return initPromise;
}

// Trigger initial load
ensureReady();

async function callHandler(channel, ...args) {
  await ensureReady();
  const handler = ipcHandlers[channel];
  if (!handler) {
    throw new Error(`Handler for ${channel} not found`);
  }
  return handler({}, ...args);
}

const engine = {
  search: (query, light = false) => callHandler('music:search', query, light),
  searchPage: (query, filter) => callHandler('music:searchPage', query, filter),
  searchMore: (token, filter) => callHandler('music:searchMore', token, filter),
  suggestions: (query) => callHandler('music:suggestions', query),
  getAlbum: (id) => callHandler('music:album', id),
  getArtist: (id) => callHandler('music:artist', id),
  getPlaylist: (id) => callHandler('music:playlist', id),
  getPlaylistPage: (id) => callHandler('music:playlistPage', id),
  getHome: () => callHandler('music:home'),
  getCharts: () => callHandler('music:charts'),
  getMoods: () => callHandler('music:moods'),
  marketInfo: () => callHandler('market:current'),
  getLyrics: (track) => callHandler('music:lyrics', track),
  enrichDurations: (ids) => callHandler('music:enrich', ids),
  getCanvas: (q) => callHandler('canvas:get', q),
  handleStream: async (req, res) => {
    await ensureReady();
    if (!streamHandler) {
      res.writeHead(503, { 'Content-Type': 'text/plain' });
      res.end('Stream handler not ready');
      return;
    }
    return streamHandler(req, res);
  },
};

module.exports = engine;
