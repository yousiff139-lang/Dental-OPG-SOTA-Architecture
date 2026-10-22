const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const backendDir = path.resolve(__dirname, '..', 'backend');

let cmd;
let args;

const venvWin = path.join(backendDir, '.venv', 'Scripts', 'python.exe');
const venvPosix = path.join(backendDir, '.venv', 'bin', 'python');

if (fs.existsSync(venvWin)) {
  cmd = venvWin;
  args = ['-m', 'uvicorn', 'app.main:app', '--reload', '--host', '0.0.0.0', '--port', '8000'];
} else if (fs.existsSync(venvPosix)) {
  cmd = venvPosix;
  args = ['-m', 'uvicorn', 'app.main:app', '--reload', '--host', '0.0.0.0', '--port', '8000'];
} else {
  let hasUv = false;
  try {
    execSync('uv --version', { stdio: 'ignore' });
    hasUv = true;
  } catch {}

  if (hasUv) {
    cmd = 'uv';
    args = ['run', 'uvicorn', 'app.main:app', '--reload', '--host', '0.0.0.0', '--port', '8000'];
  } else {
    cmd = process.platform === 'win32' ? 'python' : 'python3';
    args = ['-m', 'uvicorn', 'app.main:app', '--reload', '--host', '0.0.0.0', '--port', '8000'];
  }
}

console.log(`[BACKEND] Starting FastAPI using: ${cmd}`);

const child = spawn(cmd, args, {
  cwd: backendDir,
  stdio: 'inherit',
  shell: cmd === 'uv' || !cmd.includes(path.sep),
});

child.on('error', (err) => {
  console.error('[BACKEND] Failed to start server:', err);
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
