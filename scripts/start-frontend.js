const { spawn, execSync } = require('child_process');
const path = require('path');

const frontendDir = path.resolve(__dirname, '..', 'frontend');

let pm = 'npm';
try {
  execSync('pnpm --version', { stdio: 'ignore' });
  pm = 'pnpm';
} catch {}

console.log(`[FRONTEND] Starting Next.js app with ${pm}...`);

const child = spawn(pm, ['run', 'dev'], {
  cwd: frontendDir,
  stdio: 'inherit',
  shell: true,
});

child.on('error', (err) => {
  console.error('[FRONTEND] Failed to start:', err);
  process.exit(1);
});

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
  } else {
    process.exit(code ?? 0);
  }
});

['SIGINT', 'SIGTERM'].forEach((sig) => {
  process.on(sig, () => {
    try {
      child.kill(sig);
    } catch {}
  });
});
