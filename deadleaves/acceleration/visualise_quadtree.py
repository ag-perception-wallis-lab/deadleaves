"""
Quadtree sampling visualiser for dead leaves.

Records per-step snapshots during ``generate_segmentation`` and writes a
self-contained HTML file that lets you scrub through the sampling history,
seeing the segmentation map, quadtree tile states, and per-leaf AABB at
every step.

Usage
-----
    from deadleaves import LeafGeometryGenerator
    from visualise_quadtree import record_and_visualise

    shape_params = {
        "area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}
    }
    model = LeafGeometryGenerator("circular", shape_params, (128, 128))
    leaf_table, seg_map = record_and_visualise(model, "quadtree_vis.html")
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .quadtree import CoverageQuadTree, leaf_aabb, Node

__all__ = ["record_and_visualise"]


def _collect_tiles(root: Node, H: int, W: int) -> list[dict]:
    """Flatten all leaf nodes into serialisable dicts."""
    tiles = []

    def _walk(node: Node):
        if node.is_leaf:
            y1c = min(node.y1, H)
            x1c = min(node.x1, W)
            if node.y0 < y1c and node.x0 < x1c:
                tiles.append(
                    {
                        "y0": node.y0,
                        "x0": node.x0,
                        "y1": y1c,
                        "x1": x1c,
                        "alive": node.alive,
                    }
                )
            return
        if node.children:
            for c in node.children:
                _walk(c)

    _walk(root)
    return tiles


def record_and_visualise(
    model,
    output_path: str | Path = "quadtree_vis.html",
    *,
    snapshot_interval: int = 1,
) -> tuple:
    """Run ``generate_segmentation`` with history recording and write HTML.

    Args:
        model:
            A ``LeafGeometryGenerator`` instance (already constructed).
        output_path:
            Where to write the self-contained HTML visualiser.
        snapshot_interval:
            Record a snapshot every *n* placed leaves (1 = every leaf).

    Returns:
        ``(leaf_table, segmentation_map)`` — same as
        ``model.generate_segmentation()``.
    """
    import pandas as pd

    H, W = model.image_shape
    segmentation_map = torch.zeros(H, W, device=model.device, dtype=torch.int)
    leaf_idx = 1
    leaves_params = []

    qtree = CoverageQuadTree(model.image_shape, model.position_mask, segmentation_map)

    # History accumulator
    snapshots: list[dict] = []

    # Initial snapshot (empty canvas)
    snapshots.append(
        {
            "step": 0,
            "leaf_idx": 0,
            "coverage": 0.0,
            "aabb": None,
            "tiles": _collect_tiles(qtree._root, H, W),
            "seg": segmentation_map.cpu().flatten().tolist(),
        }
    )

    total_px = int((model.position_mask == 1).sum().item())

    while qtree.has_live_nodes:
        params = model._sample_parameters()

        if not model.position_mask[
            params["y_pos"].to(torch.int), params["x_pos"].to(torch.int)
        ]:
            continue

        y_min, x_min, y_max, x_max = leaf_aabb(params, model.leaf_shape)
        y_min = max(y_min, 0)
        x_min = max(x_min, 0)
        y_max = min(y_max, H)
        x_max = min(x_max, W)
        if y_min >= y_max or x_min >= x_max:
            continue

        live_tiles = qtree.query_live_tiles(y_min, x_min, y_max, x_max)
        if not live_tiles:
            continue

        sub_X = model.X[y_min:y_max, x_min:x_max]
        sub_Y = model.Y[y_min:y_max, x_min:x_max]
        try:
            leaf_mask = model.generate_leaf_mask((sub_X, sub_Y), params)
        except ValueError:
            continue

        sub_seg = segmentation_map[y_min:y_max, x_min:x_max]
        mask = leaf_mask & (sub_seg == 0)
        if mask.sum() > 0:
            sub_seg[mask] = leaf_idx
            leaves_params.append(params)

            for tile in live_tiles:
                qtree.update_tile(tile)

            covered = int(
                ((segmentation_map != 0) & (model.position_mask == 1)).sum().item()
            )
            coverage = covered / total_px if total_px > 0 else 1.0

            if leaf_idx % snapshot_interval == 0 or not qtree.has_live_nodes:
                snapshots.append(
                    {
                        "step": leaf_idx,
                        "leaf_idx": leaf_idx,
                        "coverage": round(coverage, 6),
                        "aabb": [y_min, x_min, y_max, x_max],
                        "tiles": _collect_tiles(qtree._root, H, W),
                        "seg": segmentation_map.cpu().flatten().tolist(),
                    }
                )

            leaf_idx += 1

        if (model.n_sample is not None) and leaf_idx >= model.n_sample:
            break

    leaf_table = pd.DataFrame(leaves_params, columns=model.params)
    leaf_table["leaf_idx"] = torch.tensor(range(leaf_idx - 1)) + 1
    leaf_table["leaf_shape"] = model.leaf_shape

    # This might break for very long trajectories; either increase step size or reduce image size
    _write_html(snapshots, H, W, Path(output_path))
    print(f"Wrote {len(snapshots)} snapshots to {output_path}")

    return leaf_table, segmentation_map


def _write_html(snapshots: list[dict], H: int, W: int, path: Path) -> None:
    data_json = json.dumps({"H": H, "W": W, "snapshots": snapshots})
    html = _HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    path.write_text(html, encoding="utf-8")


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dead leaves — quadtree sampling visualiser</title>
<style>
  :root {
    --bg: #0f1114; --bg2: #1a1d22; --fg: #d4d4d8;
    --fg2: #71717a; --accent: #60a5fa; --border: #27272a;
    --live: rgba(59,139,212,0.40); --dead: rgba(29,158,117,0.30);
    --aabb: rgba(216,90,48,0.75); --queried: rgba(226,75,74,0.5);
  }
  * { box-sizing: border-box; margin: 0; }
  body {
    font-family: -apple-system, "Segoe UI", Helvetica, sans-serif;
    background: var(--bg); color: var(--fg);
    display: flex; flex-direction: column; align-items: center;
    padding: 24px 16px; min-height: 100vh;
  }
  h1 { font-size: 18px; font-weight: 500; margin-bottom: 4px; }
  .sub { font-size: 13px; color: var(--fg2); margin-bottom: 20px; }
  .grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 16px; width: 100%; max-width: 800px;
  }
  .panel {
    background: var(--bg2); border: 0.5px solid var(--border);
    border-radius: 10px; padding: 14px; display: flex;
    flex-direction: column; align-items: center;
  }
  .panel-title {
    font-size: 13px; font-weight: 500; color: var(--fg2);
    margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;
  }
  canvas {
    width: 100%; aspect-ratio: 1; border-radius: 6px;
    image-rendering: pixelated;
  }
  .stats {
    font-size: 12px; color: var(--fg2); margin-top: 10px;
    text-align: center; line-height: 1.6;
  }
  .controls {
    width: 100%; max-width: 800px; margin-top: 20px;
    display: flex; flex-direction: column; gap: 10px;
  }
  .slider-row {
    display: flex; align-items: center; gap: 12px;
  }
  .slider-row label { font-size: 13px; color: var(--fg2); min-width: 40px; }
  .slider-row input[type=range] { flex: 1; accent-color: var(--accent); }
  .slider-row .val {
    font-size: 13px; font-weight: 500; min-width: 64px; text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .btn-row { display: flex; gap: 8px; justify-content: center; }
  .btn {
    font-size: 13px; padding: 6px 16px; border-radius: 6px; cursor: pointer;
    background: transparent; color: var(--fg); border: 0.5px solid var(--border);
    transition: background 0.15s;
  }
  .btn:hover { background: var(--border); }
  .btn:active { transform: scale(0.97); }
  .legend {
    display: flex; gap: 14px; flex-wrap: wrap; justify-content: center;
    margin-top: 16px; font-size: 12px; color: var(--fg2);
  }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .swatch { width: 10px; height: 10px; border-radius: 2px; }
  .coverage-bar-wrap {
    width: 100%; max-width: 800px; margin-top: 12px;
  }
  .coverage-bar-bg {
    width: 100%; height: 6px; background: var(--border); border-radius: 3px;
    overflow: hidden;
  }
  .coverage-bar-fill {
    height: 100%; background: var(--accent); border-radius: 3px;
    transition: width 0.05s;
  }
  .coverage-label {
    font-size: 12px; color: var(--fg2); margin-top: 4px; text-align: center;
  }
</style>
</head>
<body>
<h1>Quadtree sampling visualiser</h1>
<p class="sub">Scrub through the dead leaves sampling history</p>

<div class="grid">
  <div class="panel">
    <div class="panel-title">Segmentation map</div>
    <canvas id="seg"></canvas>
    <div class="stats" id="seg-stats"></div>
  </div>
  <div class="panel">
    <div class="panel-title">Quadtree tiles</div>
    <canvas id="qt"></canvas>
    <div class="stats" id="qt-stats"></div>
  </div>
</div>

<div class="coverage-bar-wrap">
  <div class="coverage-bar-bg"><div class="coverage-bar-fill" id="cov-fill"></div></div>
  <div class="coverage-label" id="cov-label">Coverage: 0%</div>
</div>

<div class="controls">
  <div class="slider-row">
    <label>Step</label>
    <input type="range" id="step-slider" min="0" max="0" value="0">
    <span class="val" id="step-val">0 / 0</span>
  </div>
  <div class="btn-row">
    <button class="btn" id="play-btn">Play ▶</button>
    <button class="btn" id="first-btn">⏮ First</button>
    <button class="btn" id="last-btn">Last ⏭</button>
  </div>
</div>

<div class="legend">
  <div class="legend-item"><div class="swatch" style="background:rgba(59,139,212,0.55)"></div>Live tile</div>
  <div class="legend-item"><div class="swatch" style="background:rgba(29,158,117,0.45)"></div>Dead tile</div>
  <div class="legend-item"><div class="swatch" style="background:rgba(216,90,48,0.75)"></div>Leaf AABB</div>
</div>

<script>
const DATA = __DATA_JSON__;
const { H, W, snapshots } = DATA;

const segCanvas = document.getElementById('seg');
const qtCanvas  = document.getElementById('qt');
const segCtx = segCanvas.getContext('2d');
const qtCtx  = qtCanvas.getContext('2d');
const segStats = document.getElementById('seg-stats');
const qtStats  = document.getElementById('qt-stats');
const slider   = document.getElementById('step-slider');
const stepVal  = document.getElementById('step-val');
const playBtn  = document.getElementById('play-btn');
const firstBtn = document.getElementById('first-btn');
const lastBtn  = document.getElementById('last-btn');
const covFill  = document.getElementById('cov-fill');
const covLabel = document.getElementById('cov-label');

segCanvas.width = W; segCanvas.height = H;
qtCanvas.width  = W; qtCanvas.height  = H;

slider.max = snapshots.length - 1;

// Colour palette for leaves (golden angle hue spacing)
const palette = [];
for (let i = 0; i < 4000; i++) {
  const h = (i * 137.508) % 360;
  palette.push(hslToRgb(h, 55, 58));
}

function hslToRgb(h, s, l) {
  s /= 100; l /= 100;
  const k = n => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = n => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [Math.round(f(0)*255), Math.round(f(8)*255), Math.round(f(4)*255)];
}

function renderFrame(idx) {
  const snap = snapshots[idx];

  // Segmentation
  const img = segCtx.createImageData(W, H);
  for (let i = 0; i < H * W; i++) {
    const v = snap.seg[i];
    const p = i * 4;
    if (v === 0) {
      img.data[p] = 20; img.data[p+1] = 20; img.data[p+2] = 24; img.data[p+3] = 255;
    } else {
      const [r, g, b] = palette[(v - 1) % palette.length];
      img.data[p] = r; img.data[p+1] = g; img.data[p+2] = b; img.data[p+3] = 255;
    }
  }
  segCtx.putImageData(img, 0, 0);

  // Quadtree tiles
  qtCtx.fillStyle = '#14161a';
  qtCtx.fillRect(0, 0, W, H);
  let live = 0, dead = 0;
  for (const t of snap.tiles) {
    const tw = t.x1 - t.x0, th = t.y1 - t.y0;
    qtCtx.fillStyle = t.alive ? 'rgba(59,139,212,0.35)' : 'rgba(29,158,117,0.25)';
    qtCtx.fillRect(t.x0, t.y0, tw, th);
    qtCtx.strokeStyle = 'rgba(255,255,255,0.1)';
    qtCtx.lineWidth = 0.5;
    qtCtx.strokeRect(t.x0 + 0.25, t.y0 + 0.25, tw - 0.5, th - 0.5);
    if (t.alive) live++; else dead++;
  }

  // AABB overlay
  if (snap.aabb) {
    const [yMin, xMin, yMax, xMax] = snap.aabb;
    qtCtx.strokeStyle = 'rgba(216,90,48,0.75)';
    qtCtx.lineWidth = 1.5;
    qtCtx.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin);
  }

  // Stats
  segStats.textContent = `Leaves placed: ${snap.leaf_idx}`;
  qtStats.textContent = `Live: ${live}  ·  Dead: ${dead}`;
  stepVal.textContent = `${idx} / ${snapshots.length - 1}`;

  const pct = (snap.coverage * 100).toFixed(1);
  covFill.style.width = pct + '%';
  covLabel.textContent = `Coverage: ${pct}%`;
}

slider.addEventListener('input', () => renderFrame(+slider.value));
firstBtn.addEventListener('click', () => { slider.value = 0; renderFrame(0); });
lastBtn.addEventListener('click', () => {
  slider.value = snapshots.length - 1;
  renderFrame(snapshots.length - 1);
});

let playing = false;
let playTimer = null;
playBtn.addEventListener('click', () => {
  if (playing) {
    clearInterval(playTimer);
    playBtn.textContent = 'Play ▶';
    playing = false;
  } else {
    playing = true;
    playBtn.textContent = 'Pause ⏸';
    playTimer = setInterval(() => {
      let v = +slider.value + 1;
      if (v >= snapshots.length) { v = 0; }
      slider.value = v;
      renderFrame(v);
    }, 80);
  }
});

renderFrame(0);
</script>
</body>
</html>"""
