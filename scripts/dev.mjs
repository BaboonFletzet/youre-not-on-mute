import { spawn } from 'node:child_process';
import { createServer } from 'vite';
const vite = await createServer();
await vite.listen();
vite.printUrls();
const child = spawn('node_modules/.bin/electron', ['.'], {
  stdio: 'inherit', env: { ...process.env, RML_DEV_URL: 'http://127.0.0.1:5173' },
});
child.on('exit', async code => { await vite.close(); process.exit(code ?? 0); });
process.on('SIGINT', () => child.kill('SIGTERM'));
process.on('SIGTERM', () => child.kill('SIGTERM'));
