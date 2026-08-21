const SIZE = 28;
const CELL = 10; // canvas is 280x280, so each cell is 10px
const BACKEND_URL = 'http://localhost:5000/predict'; // <-- change to your endpoint

const canvas = document.getElementById('grid');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const submitBtn = document.getElementById('submitBtn');

const BRUSH_RADIUS = 1.3; // in cells, controls stroke thickness
const BRUSH_STRENGTH = 0.85; // max intensity added at the very center per dab

let grid = Array.from({ length: SIZE }, () => Array(SIZE).fill(0));
let isDrawing = false;
let lastPos = null;

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      const v = grid[r][c]; // 0..1
      const shade = Math.round(10 + v * (234 - 10)); // 0a -> ea range to match theme
      ctx.fillStyle = `rgb(${shade},${shade},${shade + (v > 0.02 ? 6 : 0)})`;
      ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
    }
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  for (let i = 0; i <= SIZE; i++) {
    ctx.beginPath();
    ctx.moveTo(i * CELL, 0);
    ctx.lineTo(i * CELL, canvas.height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i * CELL);
    ctx.lineTo(canvas.width, i * CELL);
    ctx.stroke();
  }
}

function getEventPos(e) {
  const rect = canvas.getBoundingClientRect();
  const touch = e.touches && e.touches[0];
  const clientX = touch ? touch.clientX : e.clientX;
  const clientY = touch ? touch.clientY : e.clientY;
  return { x: clientX - rect.left, y: clientY - rect.top };
}

// Paints a soft round dab centered at (colF, rowF) in cell-space,
// using a gaussian-ish falloff so edges are antialiased like real MNIST digits.
function stampAt(colF, rowF) {
  const rMin = Math.max(0, Math.floor(rowF - BRUSH_RADIUS - 1));
  const rMax = Math.min(SIZE - 1, Math.ceil(rowF + BRUSH_RADIUS + 1));
  const cMin = Math.max(0, Math.floor(colF - BRUSH_RADIUS - 1));
  const cMax = Math.min(SIZE - 1, Math.ceil(colF + BRUSH_RADIUS + 1));

  for (let r = rMin; r <= rMax; r++) {
    for (let c = cMin; c <= cMax; c++) {
      const dx = c + 0.5 - colF;
      const dy = r + 0.5 - rowF;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > BRUSH_RADIUS + 1) continue;
      const falloff = Math.max(0, 1 - dist / (BRUSH_RADIUS + 1));
      const add = BRUSH_STRENGTH * falloff * falloff;
      grid[r][c] = Math.min(1, grid[r][c] + add);
    }
  }
}

function paintFromEvent(e) {
  const { x, y } = getEventPos(e);
  const colF = x / CELL;
  const rowF = y / CELL;

  if (lastPos) {
    // interpolate between last point and this one so fast strokes stay smooth
    const steps = Math.ceil(Math.hypot(colF - lastPos.c, rowF - lastPos.r) / 0.5) || 1;
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      stampAt(lastPos.c + (colF - lastPos.c) * t, lastPos.r + (rowF - lastPos.r) * t);
    }
  } else {
    stampAt(colF, rowF);
  }

  lastPos = { c: colF, r: rowF };
  draw();
}

canvas.addEventListener('mousedown', (e) => { isDrawing = true; lastPos = null; paintFromEvent(e); });
canvas.addEventListener('mousemove', (e) => { if (isDrawing) paintFromEvent(e); });
window.addEventListener('mouseup', () => { isDrawing = false; lastPos = null; });

canvas.addEventListener('touchstart', (e) => { e.preventDefault(); isDrawing = true; lastPos = null; paintFromEvent(e); });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); if (isDrawing) paintFromEvent(e); });
window.addEventListener('touchend', () => { isDrawing = false; lastPos = null; });

document.getElementById('clearBtn').addEventListener('click', () => {
  grid = Array.from({ length: SIZE }, () => Array(SIZE).fill(0));
  draw();
  setStatus('', '');
});

submitBtn.addEventListener('click', async () => {
  const pixels = grid.flat().map((v) => Math.round(v * 1000) / 1000);
  const payload = { width: SIZE, height: SIZE, pixels };

  submitBtn.disabled = true;
  setStatus('Sending…', '');

  try {
    const res = await fetch(BACKEND_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error(`Server responded ${res.status}`);

    setStatus('Sent successfully.', 'success');
  } catch (err) {
    setStatus(`Failed: ${err.message}`, 'error');
  } finally {
    submitBtn.disabled = false;
  }
});

function setStatus(text, type) {
  statusEl.textContent = text;
  statusEl.className = 'status' + (type ? ' ' + type : '');
}

draw();