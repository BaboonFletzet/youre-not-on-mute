const { app, BrowserWindow, ipcMain, desktopCapturer, session, dialog, systemPreferences, shell } = require('electron');
const { spawn } = require('node:child_process');
const { randomBytes } = require('node:crypto');
const path = require('node:path');
const fs = require('node:fs');
const net = require('node:net');

const root = path.resolve(__dirname, '..');
const token = randomBytes(32).toString('hex');
let mainWindow, python, port, selectedSource;
let availableSources = new Map();
const devUrl = process.env.RML_DEV_URL;
const brandName = "You're NOT on Mute!";
app.setName(brandName);

function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer(); s.on('error', reject);
    s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => resolve(p)); });
  });
}

app.whenReady().then(async () => {
  port = await freePort();
  python = spawn(path.join(root, '.venv/bin/python'), ['-m', 'services.server'], {
    cwd: root, stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, RML_TOKEN: token, RML_PORT: String(port),
      MPLCONFIGDIR: path.join(root, '.cache/matplotlib'), PYTHONUNBUFFERED: '1' }
  });
  python.stdout.on('data', b => process.stdout.write(b));
  python.stderr.on('data', b => process.stderr.write(b));
  python.on('error', e => dialog.showErrorBox('Local service could not start', e.message));
  python.on('exit', code => { if (code) console.error(`Local service exited (${code}).`); });

  mainWindow = new BrowserWindow({ width: 1330, height: 900, minWidth: 1000, minHeight: 740,
    title: brandName, backgroundColor: '#f5f6f7', titleBarStyle: 'hiddenInset',
    webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true,
      nodeIntegration: false, sandbox: true, backgroundThrottling: false } });
  mainWindow.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  mainWindow.webContents.on('will-navigate', (event) => event.preventDefault());
  const trusted = event => event.sender === mainWindow.webContents;
  ipcMain.handle('config', event => {
    if (!trusted(event)) throw new Error('Untrusted sender');
    return { token, baseUrl: `http://127.0.0.1:${port}`, streamUrl: `ws://127.0.0.1:${port}/stream` };
  });
  ipcMain.handle('sources', async event => {
    if (!trusted(event)) throw new Error('Untrusted sender');
    if (process.platform === 'darwin' && systemPreferences.getMediaAccessStatus('screen') === 'denied') {
      throw new Error('SCREEN_PERMISSION_DENIED: Enable Electron in macOS Screen & System Audio Recording settings, then restart the app.');
    }
    const sources = await desktopCapturer.getSources({ types: ['window', 'screen'], thumbnailSize: { width: 320, height: 200 } });
    availableSources = new Map(sources.filter(s => s.name !== brandName).map(source => [source.id, source]));
    return [...availableSources.values()].map(s => ({ id: s.id, name: s.name, thumbnail: s.thumbnail.toDataURL() }));
  });
  ipcMain.handle('capture-settings', event => {
    if (!trusted(event)) throw new Error('Untrusted sender');
    if (process.platform === 'darwin') return shell.openExternal('x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture');
  });
  ipcMain.handle('sample', event => {
    if (!trusted(event)) throw new Error('Untrusted sender');
    return fs.readFileSync(path.join(root, '.cache/samples/speech.mp4'));
  });
  ipcMain.handle('select-source', async (event, id) => {
    if (!trusted(event) || typeof id !== 'string') throw new Error('Invalid capture request');
    selectedSource = availableSources.get(id);
    if (!selectedSource) throw new Error('This window is no longer available. Reopen the source picker and try again.');
    return true;
  });
  session.defaultSession.setPermissionCheckHandler((contents, permission, requestingOrigin, details) => {
    const trustedWindow = contents === mainWindow.webContents;
    // Electron 36 performs these media checks for getDisplayMedia before its
    // legacy empty-mediaTypes request identifies the operation as screen capture.
    const allowed = trustedWindow && (
      permission === 'display-capture' ||
      (permission === 'media' && ['video', 'audio'].includes(details?.mediaType))
    );
    if (permission === 'media' || permission === 'display-capture') {
      console.log(`[capture] permission check: ${permission}, trustedWindow=${trustedWindow}, mediaType=${details?.mediaType ?? 'n/a'}, allowed=${allowed}`);
    }
    return allowed;
  });
  session.defaultSession.setPermissionRequestHandler((contents, permission, callback, details) => {
    const trustedWindow = contents === mainWindow.webContents;
    // Electron 36 represents getDisplayMedia as `media` with an empty list.
    // Camera and microphone requests contain video/audio entries and stay denied.
    const legacyScreenRequest = permission === 'media' && Array.isArray(details?.mediaTypes) && details.mediaTypes.length === 0;
    const allowed = trustedWindow && (permission === 'display-capture' || legacyScreenRequest);
    console.log(`[capture] permission request: ${permission}, trustedWindow=${trustedWindow}, mediaTypes=${details?.mediaTypes?.join(',') ?? 'n/a'}, allowed=${allowed}`);
    callback(allowed);
  });
  session.defaultSession.setDisplayMediaRequestHandler((request, callback) => {
    if (!selectedSource) {
      console.error('[capture] display request arrived without a selected source');
      return callback({});
    }
    const source = selectedSource; selectedSource = undefined;
    console.log(`[capture] providing selected source: ${source.name}`);
    callback({ video: source });
  });
  if (devUrl) await mainWindow.loadURL(devUrl);
  else await mainWindow.loadFile(path.join(root, 'dist/index.html'));
  if (process.env.RML_SCREENSHOT) {
    setTimeout(async () => {
      const image = await mainWindow.webContents.capturePage();
      fs.writeFileSync(process.env.RML_SCREENSHOT, image.toPNG());
    }, 6000);
  }
});
app.on('window-all-closed', () => app.quit());
app.on('before-quit', () => { python?.kill('SIGTERM'); });
