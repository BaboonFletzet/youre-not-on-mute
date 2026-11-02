// Real desktop smoke test. Uses the actual local model and optional downloaded sample.
import { _electron as electron } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { createServer } from 'vite';
fs.mkdirSync('.cache/screenshots', { recursive: true });
const vite = process.argv.includes('--dev') ? await createServer() : null;
if (vite) await vite.listen();
const app = await electron.launch({ args: ['.'], timeout: 30000,
  env: { ...process.env, ...(vite ? { RML_DEV_URL: 'http://127.0.0.1:5173' } : {}) } });
try {
  const page = await app.firstWindow();
  const permission = await app.evaluate(({ systemPreferences }) => systemPreferences.getMediaAccessStatus('screen'));
  console.log('Screen capture permission:', permission);
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.getByRole('heading', { name: 'Read visible speech.' }).waitFor();
  await page.getByText('READY', { exact: true }).waitFor({ timeout: 60000 });
  await page.screenshot({ path: '.cache/screenshots/workspace.png', fullPage: true });
  if (permission === 'denied') {
    await page.getByRole('button', { name: 'Choose a window', exact: true }).click();
    await page.getByRole('button', { name: /Open Screen.*settings/ }).waitFor();
    console.log('Capture permission guidance verified.');
  }
  if (process.argv.includes('--streaming')) await page.locator('input[type="file"]').setInputFiles('.cache/samples/streaming.mp4');
  else await page.getByText('Try a sample clip →').click();
  await page.getByLabel('Show what the model sees').check();
  await page.getByRole('button', { name: 'Start lip reading' }).click();
  try { await page.locator('.caption-list article').first().waitFor({ timeout: 45000 }); }
  catch (e) { console.error('Visible status:', await page.locator('.notice').allTextContents()); throw e; }
  console.log('REAL MODEL CAPTION:', await page.locator('.caption-list').innerText());
  await page.getByText('COMPLETE', { exact: true }).waitFor({ timeout: 45000 });
  if (process.argv.includes('--streaming')) {
    await page.waitForFunction(() => document.querySelectorAll('.caption-list article:not(.provisional)').length >= 4, undefined, { timeout: 45000 });
  }
  console.log('FINAL MODEL CAPTION:', await page.locator('.caption-list').innerText());
  await page.getByAltText('Eight grayscale mouth frames from the model input').waitFor({ timeout: 30000 });
  await page.screenshot({ path: '.cache/screenshots/recognition.png', fullPage: true });
  await page.getByRole('button', { name: 'Stop reading' }).click();
  await page.getByRole('button', { name: 'Start lip reading' }).waitFor({ timeout: 45000 });
  await page.getByRole('button', { name: 'How it works' }).click();
  await page.getByRole('heading', { name: 'Using the app' }).waitFor();
  await page.screenshot({ path: '.cache/screenshots/help.png', fullPage: true });
  await page.getByRole('button', { name: 'Close help' }).click();
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setContentSize(1000, 740));
  await page.waitForFunction(() => window.innerWidth <= 1000);
  const overflows = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  if (overflows) throw new Error('Workspace overflows horizontally at minimum window size');
  await page.screenshot({ path: '.cache/screenshots/compact.png', fullPage: true });
  if (process.argv.includes('--capture') && permission === 'granted') {
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setContentSize(1330, 900));
    const fixturePath = path.resolve('.cache/samples/capture-fixture.html');
    fs.writeFileSync(fixturePath, '<!doctype html><html><head><title>Mute Lab Capture Fixture</title></head><body style="margin:0;background:#000;display:grid;place-items:center;height:100vh"><video style="width:100%;height:100%;object-fit:contain" src="speech.mp4" autoplay muted loop playsinline></video></body></html>');
    const fixtureId = await app.evaluate(async ({ BrowserWindow }, filename) => {
      const fixture = new BrowserWindow({ width: 640, height: 480, title: 'Mute Lab Capture Fixture', webPreferences: { sandbox: true, nodeIntegration: false, contextIsolation: true } });
      await fixture.loadFile(filename);
      await fixture.webContents.executeJavaScript('document.querySelector("video").play()');
      return fixture.id;
    }, fixturePath);
    await page.getByRole('button', { name: 'Choose source', exact: true }).click();
    await page.getByRole('button', { name: 'Mute Lab Capture Fixture', exact: true }).click();
    await page.getByRole('button', { name: 'Start lip reading' }).click();
    await page.waitForFunction(() => document.querySelectorAll('.caption-list article:not(.provisional)').length >= 2, undefined, { timeout: 60000 });
    console.log('OWNED WINDOW CAPTURE CAPTIONS:', await page.locator('.caption-list').innerText());
    await page.screenshot({ path: '.cache/screenshots/window-capture.png', fullPage: true });
    await page.getByRole('button', { name: 'Stop reading' }).click();
  await page.getByRole('button', { name: 'Start lip reading' }).waitFor({ timeout: 45000 });
    await app.evaluate(({ BrowserWindow }, id) => BrowserWindow.fromId(id)?.close(), fixtureId);
    console.log('Owned-window screen capture and two completed intervals verified.');
  }
  if (errors.length) throw new Error(errors.join('\n'));
  console.log('Desktop smoke test passed.');
} finally { await app.close(); await vite?.close(); }
