#!/usr/bin/env python3
"""
RCP Analysis Launcher

Flask web app for running RCP analyses and viewing results.
All subprocesses run with analysis/ as the working directory.
Results are written to analysis/results/.

Usage:
    python analysis/launcher.py
    # Then open http://localhost:5050
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_file

# ---------------------------------------------------------------------------
# Paths (BASE_DIR = the analysis/ directory, regardless of where you launch from)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.resolve()
RUNS_DIR = BASE_DIR.parent / "runs"
RESULTS_DIR = BASE_DIR / "results"
ROOT_CONFIG = BASE_DIR.parent / "config.json"
DOMAINS_DIR = BASE_DIR.parent / "domains"

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_rel(p: Path) -> str:
    """Return p as a string relative to BASE_DIR (analysis/).
    Works correctly for paths outside BASE_DIR (e.g. ../runs)."""
    return os.path.relpath(str(p), str(BASE_DIR))


def _resolve_run(name: str) -> str:
    """Resolve a run dir name or path to a string relative to BASE_DIR."""
    if not name:
        return _to_rel(RUNS_DIR)
    p = Path(name)
    if p.is_absolute():
        return str(p)
    candidate = RUNS_DIR / name
    if candidate.is_dir():
        return _to_rel(candidate)
    candidate2 = BASE_DIR.parent / name
    if candidate2.exists():
        return _to_rel(candidate2)
    return name


def _resolve_config(name: str) -> str:
    """Resolve a config name or path to a string relative to BASE_DIR."""
    if not name:
        return _to_rel(ROOT_CONFIG)
    p = Path(name)
    if p.is_absolute():
        return str(p)
    candidate = BASE_DIR.parent / name
    if candidate.exists():
        return _to_rel(candidate)
    return name


def _collect_files(out_dir: Path) -> list:
    if not out_dir or not out_dir.is_dir():
        return []
    ext_map = {"png": "image", "json": "json", "txt": "text", "html": "html"}
    files = []
    for f in sorted(out_dir.iterdir()):
        ext = f.suffix.lstrip(".")
        if ext in ext_map:
            files.append({"name": f.name, "path": _to_rel(f), "type": ext_map[ext]})
    return files


def _build_command(analysis: str, args: dict, ts: str):
    """Return (cmd_list, output_dir_or_None)."""
    out_dir = None

    if analysis == "analyze":
        data_dir = _resolve_run(args.get("data_dir", ""))
        config = _resolve_config(args.get("config", ""))
        out_dir = RESULTS_DIR / f"analyze-{ts}"
        cmd = [sys.executable, "analyze.py",
               "--data-dir", data_dir,
               "--config", config,
               "--output-dir", _to_rel(out_dir)]
        if args.get("model"):
            cmd += ["--model", args["model"]]
        if args.get("figures_only"):
            cmd += ["--figures-only"]

    elif analysis == "validate":
        data_dir = _resolve_run(args.get("data_dir", ""))
        config = _resolve_config(args.get("config", ""))
        cmd = [sys.executable, "validate.py",
               "--data-dir", data_dir,
               "--config", config]
        tests = args.get("tests") or []
        if tests:
            cmd += ["--test"] + tests

    elif analysis == "explanations":
        # data_dir must be the project root (the script appends /runs/ internally)
        out_dir = RESULTS_DIR / f"explanations-{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "results.json"
        cmd = [sys.executable, "analyze_explanations.py",
               "..",   # project root, relative to analysis/ cwd
               "--output", _to_rel(out_file)]

    elif analysis == "permutation":
        data_dir = _resolve_run(args.get("data_dir", ""))
        config = _resolve_config(args.get("config", ""))
        out_dir = RESULTS_DIR / f"permutation-{ts}"
        cmd = [sys.executable, "permutation_tests.py",
               "--data-dir", data_dir,
               "--config", config,
               "--output-dir", _to_rel(out_dir)]
        if args.get("model"):
            cmd += ["--model", args["model"]]

    elif analysis == "cluster":
        config = _resolve_config(args.get("config", ""))
        out_dir = RESULTS_DIR / f"cluster-{ts}"
        cmd = [sys.executable, "cluster-validation/cluster_validate.py",
               "--config", config,
               "--output", _to_rel(out_dir)]
        if args.get("embed_model"):
            cmd += ["--model", args["embed_model"]]

    elif analysis == "factor":
        data_dirs = args.get("data_dirs") or []
        if not data_dirs:
            raise ValueError("Select at least one run directory.")
        config = _resolve_config(args.get("config", ""))
        out_dir = RESULTS_DIR / f"factor-{ts}"
        resolved = [_resolve_run(d) for d in data_dirs]
        cmd = ([sys.executable, "factor-validation/factor_validate.py",
                "--data"] + resolved +
               ["--config", config, "--output", _to_rel(out_dir)])
        if args.get("n_factors"):
            cmd += ["--n-factors", str(args["n_factors"])]

    else:
        raise ValueError(f"Unknown analysis: {analysis!r}")

    return cmd, out_dir


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return HTML


@app.get("/api/runs")
def api_runs():
    """List run directories that contain JSONL data files."""
    runs = []
    if RUNS_DIR.is_dir():
        for d in sorted(RUNS_DIR.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and list(d.glob("*.jsonl")):
                runs.append(d.name)
    return jsonify(runs=runs)


@app.get("/api/configs")
def api_configs():
    """List available config files."""
    configs = []
    if ROOT_CONFIG.exists():
        configs.append({"label": "config.json", "value": "config.json"})
    if DOMAINS_DIR.is_dir():
        for f in sorted(DOMAINS_DIR.glob("config*.json")):
            configs.append({"label": f"domains/{f.name}", "value": f"domains/{f.name}"})
    return jsonify(configs=configs)


@app.post("/api/run")
def api_run():
    body = request.get_json()
    analysis = body.get("analysis", "")
    args = body.get("args", {})
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
        cmd, out_dir = _build_command(analysis, args, ts)
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return jsonify(ok=False, error="Analysis timed out after 10 minutes.")

    return jsonify(
        ok=result.returncode == 0,
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
        output_dir=_to_rel(out_dir) if out_dir else None,
        files=_collect_files(out_dir),
    )


@app.get("/api/file")
def api_file():
    """Serve a result file. Restricted to files inside analysis/results/."""
    rel = request.args.get("path", "")
    target = (BASE_DIR / rel).resolve()
    if not str(target).startswith(str(RESULTS_DIR.resolve())):
        return "Forbidden", 403
    if not target.is_file():
        return "Not found", 404
    return send_file(target)


@app.get("/viewer/<name>")
def viewer(name: str):
    allowed = {
        "explorer": BASE_DIR / "explorer.html",
        "explanations": BASE_DIR / "explanation_results_viewer.html",
    }
    path = allowed.get(name)
    if not path or not path.exists():
        return "Not found", 404
    return send_file(path)


# ---------------------------------------------------------------------------
# Embedded UI
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RCP Analysis Launcher</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0c0e12;
  --surface: #14171e;
  --surface-2: #1c2029;
  --border: #2a2e3a;
  --text: #dde1ea;
  --text-dim: #6b7280;
  --text-bright: #e8ecf4;
  --accent: #60a5fa;
  --accent-dim: #2563eb;
  --green: #34d399;
  --red: #f87171;
  --amber: #fbbf24;
  --mono: 'IBM Plex Mono', monospace;
  --sans: 'IBM Plex Sans', sans-serif;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: var(--sans); background: var(--bg); color: var(--text); line-height: 1.6; min-height: 100vh; }

/* Header */
header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 16px 32px;
  display: flex;
  align-items: center;
  gap: 16px;
}
header h1 { font-family: var(--mono); font-size: 15px; font-weight: 600; color: var(--accent); letter-spacing: 0.06em; }
header .sub { font-size: 13px; color: var(--text-dim); }

/* Tab bar */
.tab-bar {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 0 32px;
  display: flex;
  overflow-x: auto;
}
.tab-btn {
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--text-dim);
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  padding: 14px 20px;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.15s;
}
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }

/* Content */
.content { max-width: 900px; margin: 0 auto; padding: 40px 32px; }

/* Panels */
.panel { display: none; }
.panel.active { display: block; }

/* Description */
.desc {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent-dim);
  border-radius: 0 6px 6px 0;
  padding: 14px 18px;
  font-size: 13px;
  color: var(--text-dim);
  margin-bottom: 28px;
  line-height: 1.7;
}
.desc strong { color: var(--text); font-weight: 500; }

/* Forms */
.form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
.field { display: flex; flex-direction: column; gap: 6px; }
.field.full { grid-column: 1 / -1; }
label.field-label {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--text-dim);
}
label.field-label .opt { font-style: italic; text-transform: none; letter-spacing: 0; font-weight: 400; }
select, input[type="text"], input[type="number"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  font-family: var(--sans);
  font-size: 13px;
  padding: 8px 12px;
  outline: none;
  transition: border-color 0.15s;
}
select:focus, input:focus { border-color: var(--accent); }
select[multiple] { height: 130px; padding: 4px; }
select[multiple] option { padding: 5px 8px; border-radius: 3px; }
select[multiple] option:checked { background: var(--accent-dim); color: #fff; }

/* Checkboxes */
.check-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.check-item {
  display: flex;
  align-items: center;
  gap: 6px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 5px 10px;
  cursor: pointer;
  font-size: 13px;
  transition: border-color 0.15s;
}
.check-item:hover { border-color: var(--accent); }
.check-item input { accent-color: var(--accent); cursor: pointer; }

/* Run button */
.run-btn {
  background: var(--accent-dim);
  color: #fff;
  border: none;
  border-radius: 6px;
  padding: 10px 28px;
  font-family: var(--sans);
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s;
  display: inline-flex;
  align-items: center;
  gap: 10px;
}
.run-btn:hover:not(:disabled) { background: var(--accent); color: #000; }
.run-btn:disabled { background: var(--border); color: var(--text-dim); cursor: not-allowed; }
.spinner {
  display: none;
  width: 14px;
  height: 14px;
  border: 2px solid rgba(255,255,255,0.3);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  flex-shrink: 0;
}
.run-btn.running .spinner { display: block; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Results */
.results { margin-top: 36px; display: none; }
.results.visible { display: block; }
.divider { border: none; border-top: 1px solid var(--border); margin-bottom: 24px; }
.results-header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
.results-header h3 {
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-bright);
}
.status-badge {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  padding: 3px 8px;
  border-radius: 3px;
}
.status-badge.ok { background: rgba(52,211,153,0.15); color: var(--green); }
.status-badge.err { background: rgba(248,113,113,0.15); color: var(--red); }
.section-label {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-dim);
  margin-bottom: 10px;
}
.log-block {
  background: #080a0e;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 16px;
  font-family: var(--mono);
  font-size: 12px;
  color: #a0aab8;
  white-space: pre-wrap;
  overflow-x: auto;
  max-height: 340px;
  overflow-y: auto;
  margin-bottom: 24px;
  line-height: 1.6;
}

/* Result files */
.file-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; margin-bottom: 20px; }
.file-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.file-name {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-dim);
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  background: var(--surface-2);
}
.file-card img { width: 100%; display: block; cursor: zoom-in; }
.file-card img:hover { opacity: 0.9; }
.json-preview {
  padding: 12px;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-dim);
  max-height: 180px;
  overflow-y: auto;
  white-space: pre-wrap;
  line-height: 1.5;
}

/* Viewer buttons */
.viewer-btns { display: flex; gap: 10px; margin-top: 4px; }
.viewer-btn {
  background: none;
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  font-family: var(--sans);
  font-size: 13px;
  padding: 8px 16px;
  cursor: pointer;
  transition: all 0.15s;
}
.viewer-btn:hover { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>

<header>
  <h1>RCP ANALYSIS LAUNCHER</h1>
  <span class="sub">Relational Consistency Probing</span>
</header>

<div class="tab-bar">
  <button class="tab-btn active" data-tab="analyze">Analyze</button>
  <button class="tab-btn" data-tab="validate">Validate</button>
  <button class="tab-btn" data-tab="explanations">Explanations</button>
  <button class="tab-btn" data-tab="permutation">Permutation Tests</button>
  <button class="tab-btn" data-tab="cluster">Cluster Validation</button>
  <button class="tab-btn" data-tab="factor">Factor Validation</button>
</div>

<div class="content">

  <!-- ── Analyze ─────────────────────────────────────────────────────────── -->
  <div class="panel active" id="panel-analyze">
    <div class="desc">
      <strong>Analyze</strong> — Runs the main RCP analysis pipeline on collected data.
      Computes similarity matrices, domain drift metrics, and generates visualizations
      (MDS projections, domain drift chart, vector decomposition).
    </div>
    <div class="form-grid">
      <div class="field">
        <label class="field-label">Data directory</label>
        <select id="analyze-data-dir" class="runs-select"></select>
      </div>
      <div class="field">
        <label class="field-label">Config</label>
        <select id="analyze-config" class="configs-select"></select>
      </div>
      <div class="field">
        <label class="field-label">Model filter <span class="opt">(optional)</span></label>
        <input type="text" id="analyze-model" placeholder="e.g. claude-sonnet">
      </div>
      <div class="field" style="justify-content: flex-end; padding-top: 20px;">
        <label class="check-item" style="width: fit-content;">
          <input type="checkbox" id="analyze-figures-only">
          Figures only (skip matrix recompute)
        </label>
      </div>
    </div>
    <button class="run-btn" data-analysis="analyze">
      <span class="spinner"></span>Run Analysis
    </button>
    <div class="results" id="results-analyze">
      <hr class="divider">
      <div class="results-header">
        <h3>Results</h3>
        <span class="status-badge" id="badge-analyze"></span>
      </div>
      <div class="section-label">Output log</div>
      <pre class="log-block" id="log-analyze"></pre>
      <div id="files-analyze"></div>
      <div class="viewer-btns">
        <button class="viewer-btn" onclick="window.open('/viewer/explorer','_blank')">Open Results Explorer →</button>
      </div>
    </div>
  </div>

  <!-- ── Validate ─────────────────────────────────────────────────────────── -->
  <div class="panel" id="panel-validate">
    <div class="desc">
      <strong>Validate</strong> — Runs methodology validation tests V1–V7 from the
      experiment protocol. Each test prints PASS/FAIL with supporting data.
      Run after data collection to verify methodology before interpreting results.
    </div>
    <div class="form-grid">
      <div class="field">
        <label class="field-label">Data directory</label>
        <select id="validate-data-dir" class="runs-select"></select>
      </div>
      <div class="field">
        <label class="field-label">Config</label>
        <select id="validate-config" class="configs-select"></select>
      </div>
      <div class="field full">
        <label class="field-label">Tests to run <span class="opt">(leave all unchecked to run all)</span></label>
        <div class="check-grid">
          <label class="check-item"><input type="checkbox" name="validate-test" value="V1"> V1</label>
          <label class="check-item"><input type="checkbox" name="validate-test" value="V2"> V2</label>
          <label class="check-item"><input type="checkbox" name="validate-test" value="V3"> V3</label>
          <label class="check-item"><input type="checkbox" name="validate-test" value="V4"> V4</label>
          <label class="check-item"><input type="checkbox" name="validate-test" value="V5"> V5</label>
          <label class="check-item"><input type="checkbox" name="validate-test" value="V6"> V6</label>
          <label class="check-item"><input type="checkbox" name="validate-test" value="V7"> V7</label>
        </div>
      </div>
    </div>
    <button class="run-btn" data-analysis="validate">
      <span class="spinner"></span>Run Validation
    </button>
    <div class="results" id="results-validate">
      <hr class="divider">
      <div class="results-header">
        <h3>Results</h3>
        <span class="status-badge" id="badge-validate"></span>
      </div>
      <div class="section-label">Output log</div>
      <pre class="log-block" id="log-validate"></pre>
    </div>
  </div>

  <!-- ── Explanations ─────────────────────────────────────────────────────── -->
  <div class="panel" id="panel-explanations">
    <div class="desc">
      <strong>Explanations</strong> — Quantifies compliance markers, preamble recall
      (ROUGE-1), and lexical patterns across all explanation probe responses.
      Scans all <code style="font-family:var(--mono);font-size:12px;">explanations.jsonl</code>
      files in the runs directory automatically.
    </div>
    <button class="run-btn" data-analysis="explanations" style="margin-bottom: 0;">
      <span class="spinner"></span>Run Analysis
    </button>
    <div class="results" id="results-explanations">
      <hr class="divider">
      <div class="results-header">
        <h3>Results</h3>
        <span class="status-badge" id="badge-explanations"></span>
      </div>
      <div class="section-label">Output log</div>
      <pre class="log-block" id="log-explanations"></pre>
      <div id="files-explanations"></div>
      <div class="viewer-btns">
        <button class="viewer-btn" onclick="window.open('/viewer/explanations','_blank')">Open Explanation Viewer →</button>
      </div>
    </div>
  </div>

  <!-- ── Permutation Tests ─────────────────────────────────────────────────── -->
  <div class="panel" id="panel-permutation">
    <div class="desc">
      <strong>Permutation Tests</strong> — Tests whether domain drift is statistically
      significant. H1: inter-domain drift exceeds intra-domain drift.
      H2: drift ordering is consistent across framings.
    </div>
    <div class="form-grid">
      <div class="field">
        <label class="field-label">Data directory</label>
        <select id="permutation-data-dir" class="runs-select"></select>
      </div>
      <div class="field">
        <label class="field-label">Config</label>
        <select id="permutation-config" class="configs-select"></select>
      </div>
      <div class="field">
        <label class="field-label">Model filter <span class="opt">(optional)</span></label>
        <input type="text" id="permutation-model" placeholder="e.g. claude-sonnet">
      </div>
    </div>
    <button class="run-btn" data-analysis="permutation">
      <span class="spinner"></span>Run Permutation Tests
    </button>
    <div class="results" id="results-permutation">
      <hr class="divider">
      <div class="results-header">
        <h3>Results</h3>
        <span class="status-badge" id="badge-permutation"></span>
      </div>
      <div class="section-label">Output log</div>
      <pre class="log-block" id="log-permutation"></pre>
      <div id="files-permutation"></div>
    </div>
  </div>

  <!-- ── Cluster Validation ────────────────────────────────────────────────── -->
  <div class="panel" id="panel-cluster">
    <div class="desc">
      <strong>Cluster Validation</strong> — Validates that domain assignments
      (physical, institutional, moral) correspond to semantic clustering in
      embedding space. Uses sentence-transformers and silhouette analysis.
      <br><span style="color: var(--amber); font-size: 12px;">Requires: sentence-transformers</span>
    </div>
    <div class="form-grid">
      <div class="field">
        <label class="field-label">Config</label>
        <select id="cluster-config" class="configs-select"></select>
      </div>
      <div class="field">
        <label class="field-label">Embedding model <span class="opt">(optional)</span></label>
        <input type="text" id="cluster-embed-model" placeholder="all-MiniLM-L6-v2">
      </div>
    </div>
    <button class="run-btn" data-analysis="cluster">
      <span class="spinner"></span>Run Cluster Validation
    </button>
    <div class="results" id="results-cluster">
      <hr class="divider">
      <div class="results-header">
        <h3>Results</h3>
        <span class="status-badge" id="badge-cluster"></span>
      </div>
      <div class="section-label">Output log</div>
      <pre class="log-block" id="log-cluster"></pre>
      <div id="files-cluster"></div>
    </div>
  </div>

  <!-- ── Factor Validation ─────────────────────────────────────────────────── -->
  <div class="panel" id="panel-factor">
    <div class="desc">
      <strong>Factor Validation</strong> — Validates domain assignments via exploratory
      factor analysis on LLM similarity rating data. Complements cluster validation
      by working from actual response structure rather than a proxy model.
    </div>
    <div class="form-grid">
      <div class="field full">
        <label class="field-label">Run directories <span class="opt">(Ctrl/Cmd+click to select multiple)</span></label>
        <select id="factor-data-dirs" class="runs-select" multiple></select>
      </div>
      <div class="field">
        <label class="field-label">Config</label>
        <select id="factor-config" class="configs-select"></select>
      </div>
      <div class="field">
        <label class="field-label">Number of factors <span class="opt">(optional)</span></label>
        <input type="number" id="factor-n-factors" placeholder="3" min="1" max="20">
      </div>
    </div>
    <button class="run-btn" data-analysis="factor">
      <span class="spinner"></span>Run Factor Validation
    </button>
    <div class="results" id="results-factor">
      <hr class="divider">
      <div class="results-header">
        <h3>Results</h3>
        <span class="status-badge" id="badge-factor"></span>
      </div>
      <div class="section-label">Output log</div>
      <pre class="log-block" id="log-factor"></pre>
      <div id="files-factor"></div>
    </div>
  </div>

</div><!-- /.content -->

<script>
// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('panel-' + tab).classList.add('active');
  });
});

// ---------------------------------------------------------------------------
// Populate run and config dropdowns on load
// ---------------------------------------------------------------------------
async function loadRuns() {
  const res = await fetch('/api/runs');
  const data = await res.json();
  const runs = data.runs;

  document.querySelectorAll('.runs-select').forEach(sel => {
    if (!sel.multiple) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '(all — let script decide)';
      sel.appendChild(opt);
    }
    runs.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    });
  });
}

async function loadConfigs() {
  const res = await fetch('/api/configs');
  const data = await res.json();
  document.querySelectorAll('.configs-select').forEach(sel => {
    data.configs.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.value;
      opt.textContent = c.label;
      sel.appendChild(opt);
    });
  });
}

// ---------------------------------------------------------------------------
// Run analysis
// ---------------------------------------------------------------------------
document.querySelectorAll('.run-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const analysis = btn.dataset.analysis;
    const args = collectArgs(analysis);
    if (args === null) return;

    const resultsEl = document.getElementById('results-' + analysis);
    const logEl     = document.getElementById('log-' + analysis);
    const badgeEl   = document.getElementById('badge-' + analysis);
    const filesEl   = document.getElementById('files-' + analysis);

    // Reset state
    if (logEl)   logEl.textContent = 'Running\u2026';
    if (badgeEl) { badgeEl.textContent = ''; badgeEl.className = 'status-badge'; }
    if (filesEl) filesEl.innerHTML = '';
    resultsEl.classList.add('visible');

    btn.disabled = true;
    btn.classList.add('running');

    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analysis, args }),
      });
      const data = await res.json();

      if (data.error) {
        if (logEl) logEl.textContent = data.error;
        if (badgeEl) { badgeEl.textContent = 'Error'; badgeEl.className = 'status-badge err'; }
        return;
      }

      const parts = [];
      if (data.stdout) parts.push(data.stdout);
      if (data.stderr) parts.push('--- stderr ---\n' + data.stderr);
      if (logEl) logEl.textContent = parts.join('\n') || '(no output)';

      if (badgeEl) {
        badgeEl.textContent = data.ok ? 'OK' : ('Exit ' + data.returncode);
        badgeEl.className   = 'status-badge ' + (data.ok ? 'ok' : 'err');
      }

      if (filesEl && data.files && data.files.length) {
        renderFiles(filesEl, data.files);
      }
    } catch (err) {
      if (logEl)   logEl.textContent = 'Request failed: ' + err.message;
      if (badgeEl) { badgeEl.textContent = 'Error'; badgeEl.className = 'status-badge err'; }
    } finally {
      btn.disabled = false;
      btn.classList.remove('running');
    }
  });
});

function collectArgs(analysis) {
  if (analysis === 'analyze') {
    return {
      data_dir:     document.getElementById('analyze-data-dir').value,
      config:       document.getElementById('analyze-config').value,
      model:        document.getElementById('analyze-model').value.trim() || null,
      figures_only: document.getElementById('analyze-figures-only').checked,
    };
  }
  if (analysis === 'validate') {
    const checked = [...document.querySelectorAll('input[name="validate-test"]:checked')].map(el => el.value);
    return {
      data_dir: document.getElementById('validate-data-dir').value,
      config:   document.getElementById('validate-config').value,
      tests:    checked.length ? checked : null,
    };
  }
  if (analysis === 'explanations') {
    return {};  // data_dir is always project root; handled in backend
  }
  if (analysis === 'permutation') {
    return {
      data_dir: document.getElementById('permutation-data-dir').value,
      config:   document.getElementById('permutation-config').value,
      model:    document.getElementById('permutation-model').value.trim() || null,
    };
  }
  if (analysis === 'cluster') {
    return {
      config:      document.getElementById('cluster-config').value,
      embed_model: document.getElementById('cluster-embed-model').value.trim() || null,
    };
  }
  if (analysis === 'factor') {
    const sel = document.getElementById('factor-data-dirs');
    const selected = [...sel.selectedOptions].map(o => o.value);
    if (!selected.length) {
      alert('Select at least one run directory.');
      return null;
    }
    const nf = document.getElementById('factor-n-factors').value;
    return {
      data_dirs: selected,
      config:    document.getElementById('factor-config').value,
      n_factors: nf ? parseInt(nf, 10) : null,
    };
  }
  return {};
}

// ---------------------------------------------------------------------------
// Render result files (images + JSON previews)
// ---------------------------------------------------------------------------
function renderFiles(container, files) {
  const images = files.filter(f => f.type === 'image');
  const jsons  = files.filter(f => f.type === 'json');

  if (images.length) {
    appendLabel(container, 'Figures');
    const grid = document.createElement('div');
    grid.className = 'file-grid';
    images.forEach(f => {
      const card = makeCard(f.name);
      const img  = document.createElement('img');
      img.src    = '/api/file?path=' + encodeURIComponent(f.path);
      img.alt    = f.name;
      img.addEventListener('click', () => window.open(img.src, '_blank'));
      card.appendChild(img);
      grid.appendChild(card);
    });
    container.appendChild(grid);
  }

  if (jsons.length) {
    appendLabel(container, 'Data files', '16px');
    const grid = document.createElement('div');
    grid.className = 'file-grid';
    jsons.forEach(f => {
      const card = makeCard(f.name);
      const pre  = document.createElement('div');
      pre.className = 'json-preview';
      pre.textContent = 'Loading\u2026';
      card.appendChild(pre);
      grid.appendChild(card);
      fetch('/api/file?path=' + encodeURIComponent(f.path))
        .then(r => r.text())
        .then(text => {
          try { text = JSON.stringify(JSON.parse(text), null, 2); } catch (_) {}
          const truncated = text.length > 2000;
          pre.textContent = text.slice(0, 2000) + (truncated ? '\n\u2026 (truncated)' : '');
        })
        .catch(() => { pre.textContent = '(error loading file)'; });
    });
    container.appendChild(grid);
  }
}

function makeCard(name) {
  const card = document.createElement('div');
  card.className = 'file-card';
  const header = document.createElement('div');
  header.className = 'file-name';
  header.textContent = name;
  card.appendChild(header);
  return card;
}

function appendLabel(container, text, marginTop) {
  const el = document.createElement('div');
  el.className = 'section-label';
  if (marginTop) el.style.marginTop = marginTop;
  el.textContent = text;
  container.appendChild(el);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
loadRuns();
loadConfigs();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    print("RCP Analysis Launcher → http://localhost:5050")
    app.run(port=5050, debug=False)
