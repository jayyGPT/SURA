from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit_output"
OUT.mkdir(exist_ok=True)

ACTIVE_TEXT_ROOTS = [
    ROOT / "README.md",
    ROOT / "models",
    ROOT / "train",
    ROOT / "benchmarks",
    ROOT / "data",
    ROOT / "docs",
    ROOT / "paper",
    ROOT / "tools",
]
SKIP_DIRS = {".git", "audit_output", "__pycache__", ".pytest_cache", ".mypy_cache"}
TEXT_SUFFIXES = {
    ".py", ".md", ".tex", ".bib", ".txt", ".yaml", ".yml", ".json",
    ".csv", ".toml", ".ini", ".cfg", ".rst", ".sh", ".gitignore",
}


@dataclass
class Finding:
    severity: str
    category: str
    title: str
    evidence: str
    recommendation: str


@dataclass
class ClaimRow:
    claim_id: str
    paper_claim: str
    paper_location: str
    status: str
    evidence: str
    notes: str


findings: list[Finding] = []
claims: list[ClaimRow] = []


def add_finding(severity: str, category: str, title: str, evidence: str, recommendation: str) -> None:
    findings.append(Finding(severity, category, title, evidence, recommendation))


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def iter_files(base: Path = ROOT) -> Iterable[Path]:
    for path in base.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {"README", "LICENSE", ".gitignore", ".gitattributes"}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def find_lines(path: Path, pattern: str, flags: int = re.IGNORECASE) -> list[tuple[int, str]]:
    if not path.exists() or not is_text(path):
        return []
    rx = re.compile(pattern, flags)
    out: list[tuple[int, str]] = []
    for idx, line in enumerate(read_text(path).splitlines(), 1):
        if rx.search(line):
            out.append((idx, line.strip()))
    return out


def fmt_hits(path: Path, hits: Sequence[tuple[int, str]], max_hits: int = 4) -> str:
    if not hits:
        return ""
    shown = [f"{rel(path)}:{line_no}: {line}" for line_no, line in hits[:max_hits]]
    if len(hits) > max_hits:
        shown.append(f"... and {len(hits) - max_hits} more hit(s)")
    return "\n".join(shown)


def grep_repo(pattern: str, roots: Sequence[Path] | None = None, flags: int = re.IGNORECASE, max_hits: int = 20) -> list[str]:
    rx = re.compile(pattern, flags)
    hits: list[str] = []
    scan_roots = roots or ACTIVE_TEXT_ROOTS
    seen: set[Path] = set()
    for root in scan_roots:
        paths = [root] if root.is_file() else list(root.rglob("*")) if root.exists() else []
        for path in paths:
            if path in seen or not path.is_file() or not is_text(path):
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            seen.add(path)
            for line_no, line in enumerate(read_text(path).splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{rel(path)}:{line_no}: {line.strip()}")
                    if len(hits) >= max_hits:
                        return hits
    return hits


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def run(cmd: Sequence[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd), cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check
    )


def add_claim(
    claim_id: str,
    paper_claim: str,
    paper_pattern: str,
    evidence_specs: Sequence[tuple[Path, str]],
    notes: str = "",
    require_all: bool = True,
) -> None:
    paper = ROOT / "paper/main.tex"
    paper_hits = find_lines(paper, paper_pattern)
    evidence_chunks: list[str] = []
    present: list[bool] = []
    for path, pattern in evidence_specs:
        hits = find_lines(path, pattern)
        present.append(bool(hits))
        if hits:
            evidence_chunks.append(fmt_hits(path, hits))
        else:
            evidence_chunks.append(f"MISSING: {rel(path)} / {pattern}")
    if not paper_hits:
        status = "PAPER_CLAIM_NOT_FOUND"
    elif evidence_specs and ((all(present) if require_all else any(present))):
        status = "SUPPORTED"
    else:
        status = "PARTIAL_OR_UNSUPPORTED"
    claims.append(
        ClaimRow(
            claim_id=claim_id,
            paper_claim=paper_claim,
            paper_location=fmt_hits(paper, paper_hits, max_hits=2) or "not found",
            status=status,
            evidence="\n".join(evidence_chunks),
            notes=notes,
        )
    )


def inventory() -> dict:
    files = list(iter_files())
    total_bytes = sum(p.stat().st_size for p in files)
    top_counts: Counter[str] = Counter()
    top_bytes: Counter[str] = Counter()
    ext_counts: Counter[str] = Counter()
    ext_bytes: Counter[str] = Counter()
    large: list[dict] = []
    hashes: defaultdict[tuple[int, str], list[str]] = defaultdict(list)
    for path in files:
        rp = path.relative_to(ROOT)
        top = rp.parts[0]
        size = path.stat().st_size
        top_counts[top] += 1
        top_bytes[top] += size
        suffix = path.suffix.lower() or "<none>"
        ext_counts[suffix] += 1
        ext_bytes[suffix] += size
        if size >= 1_000_000:
            large.append({"path": rp.as_posix(), "bytes": size})
        # Hash files up to 25 MB; this covers source and most generated artifacts without slowing the audit.
        if size <= 25_000_000:
            try:
                hashes[(size, sha256(path))].append(rp.as_posix())
            except OSError:
                pass
    duplicates = [paths for (_key, paths) in hashes.items() if len(paths) > 1]
    large.sort(key=lambda x: x["bytes"], reverse=True)
    duplicates.sort(key=lambda x: (-len(x), x[0]))
    data = {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "top_level_counts": dict(top_counts.most_common()),
        "top_level_bytes": dict(top_bytes.most_common()),
        "extension_counts": dict(ext_counts.most_common()),
        "extension_bytes": dict(ext_bytes.most_common()),
        "large_files": large,
        "duplicate_groups": duplicates,
    }
    (OUT / "inventory.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def check_python() -> None:
    proc = run([sys.executable, "-m", "compileall", "-q", "models", "train", "benchmarks", "data", "tools"])
    if proc.returncode != 0:
        add_finding("BLOCKER", "code", "Python compilation failure", proc.stdout[-4000:], "Fix syntax/import-time compilation errors before declaring the repository final.")
    else:
        add_finding("PASS", "code", "All active Python files compile", "python -m compileall completed successfully.", "None.")

    # Parse every active Python file into an AST for a second syntax check and collect top-level imports/classes/functions.
    ast_summary: dict[str, dict[str, list[str]]] = {}
    for root_name in ["models", "train", "benchmarks", "data", "tools"]:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            try:
                tree = ast.parse(read_text(path), filename=str(path))
            except SyntaxError as exc:
                add_finding("BLOCKER", "code", f"AST parse failure: {rel(path)}", str(exc), "Repair the file or remove it if obsolete.")
                continue
            ast_summary[rel(path)] = {
                "classes": [node.name for node in tree.body if isinstance(node, ast.ClassDef)],
                "functions": [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))],
                "imports": [
                    alias.name
                    for node in tree.body
                    if isinstance(node, ast.Import)
                    for alias in node.names
                ] + [
                    (node.module or "")
                    for node in tree.body
                    if isinstance(node, ast.ImportFrom)
                ],
            }
    (OUT / "python_ast_summary.json").write_text(json.dumps(ast_summary, indent=2), encoding="utf-8")


def smoke_models() -> None:
    try:
        import numpy as np
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        add_finding("MAJOR", "code", "Model smoke tests could not import dependencies", repr(exc), "Install the declared requirements and rerun the audit.")
        return

    sys.path.insert(0, str(ROOT))
    try:
        from models.wifi_heatmap import WiFiHeatmapMLP
        from models.magnetic_sequence_cnn import MagneticSequenceCNN
        from models.pdr import StepDetector
    except Exception as exc:
        add_finding("BLOCKER", "code", "Active model import failed", repr(exc), "Fix imports before finalization.")
        return

    try:
        wifi = WiFiHeatmapMLP(num_aps=11, num_cells=17)
        wifi.eval()
        with torch.no_grad():
            logits = wifi(torch.zeros(3, 11))
        shape_ok = tuple(logits.shape) == (3, 17)
        finite_ok = bool(torch.isfinite(logits).all())
        add_finding(
            "PASS" if shape_ok and finite_ok else "BLOCKER",
            "code",
            "Wi-Fi MLP smoke test",
            f"output_shape={tuple(logits.shape)}, finite={finite_ok}",
            "None." if shape_ok and finite_ok else "Reconcile the implementation with the documented M-cell output.",
        )
    except Exception as exc:
        add_finding("BLOCKER", "code", "Wi-Fi MLP smoke test failed", repr(exc), "Fix the active Wi-Fi model.")

    try:
        mag = MagneticSequenceCNN(num_features=4)
        mag.eval()
        with torch.no_grad():
            pos, ell = mag(torch.zeros(2, 84, 4))
        ok = tuple(pos.shape) == (2, 2) and tuple(ell.shape) in {(2,), (2, 1)}
        add_finding(
            "PASS" if ok else "BLOCKER",
            "code",
            "Magnetic CNN smoke test",
            f"position_shape={tuple(pos.shape)}, uncertainty_shape={tuple(ell.shape)}",
            "None." if ok else "Reconcile tensor shapes with the paper's 2-D position and scalar uncertainty claims.",
        )
    except TypeError:
        # Constructor names changed historically; introspect and retry with defaults.
        try:
            mag = MagneticSequenceCNN()
            mag.eval()
            with torch.no_grad():
                pos, ell = mag(torch.zeros(2, 84, 4))
            ok = tuple(pos.shape) == (2, 2) and tuple(ell.shape) in {(2,), (2, 1)}
            add_finding("PASS" if ok else "BLOCKER", "code", "Magnetic CNN smoke test", f"position_shape={tuple(pos.shape)}, uncertainty_shape={tuple(ell.shape)}", "None." if ok else "Reconcile output shapes.")
        except Exception as exc:
            add_finding("BLOCKER", "code", "Magnetic CNN smoke test failed", repr(exc), "Fix the active magnetic model.")
    except Exception as exc:
        add_finding("BLOCKER", "code", "Magnetic CNN smoke test failed", repr(exc), "Fix the active magnetic model.")

    try:
        detector = StepDetector()
        values = [9.81] * 20 + [11.0] + [9.81] * 20
        outputs = [bool(detector.update(v)) for v in values]
        add_finding("PASS", "code", "PDR detector executes causally", f"detections={sum(outputs)} over {len(values)} samples", "This is a smoke test only; benchmark fidelity is checked separately against constants and equations.")
    except Exception as exc:
        add_finding("BLOCKER", "code", "PDR detector smoke test failed", repr(exc), "Fix the active PDR model.")


def check_claims() -> None:
    paper = ROOT / "paper/main.tex"
    fusion = ROOT / "train/kalmannet_wifiheatmap_magneticCNN_pdr.py"
    wifi_model = ROOT / "models/wifi_heatmap.py"
    mag_model = ROOT / "models/magnetic_sequence_cnn.py"
    pdr_model = ROOT / "models/pdr.py"
    wifi_train = ROOT / "train/train_wifi_heatmap.py"
    mag_train = ROOT / "train/train_magnetic_sequence.py"
    common = ROOT / "train/common.py"

    add_claim("C01", "Wi-Fi MLP and 84-frame magnetic CNN independently produce Cartesian measurements.", r"84-frame.*magnetic.*CNN.*Cartesian", [(wifi_model, r"class\s+WiFiHeatmapMLP"), (mag_model, r"class\s+MagneticSequenceCNN")])
    add_claim("C02", "The Wi-Fi MLP uses two 256-unit hidden layers with dropout 0.3.", r"FC\(256\).*Dropout\(0\.3\)", [(wifi_model, r"Linear\([^\n]*256"), (wifi_model, r"Dropout\(0\.3\)")])
    add_claim("C03", "The Wi-Fi model produces an M-cell softmax heatmap and Cartesian expectation.", r"softmax expectation|Softmax expectation", [(wifi_model, r"softmax"), (wifi_model, r"expected|expectation|coordinates|grid")], require_all=False)
    add_claim("C04", "RSSI is clipped to [-90,-30] and affine-mapped to [0,1], with -100 dBm missing values mapping to zero.", r"clip\}\(s_\{t,i\},-90,-30\)", [(common, r"-90"), (common, r"-30"), (common, r"-100")], require_all=False)
    add_claim("C05", "Magnetic preprocessing uses norm, vertical, horizontal, and dip features from normalized acceleration.", r"m_\{N,n\}.*m_\{V,n\}|gravity-direction proxy", [(mag_train, r"magN|m_N|magnitude"), (mag_train, r"magV|vertical"), (mag_train, r"magH|horizontal"), (mag_train, r"dip|atan2")], require_all=False)
    add_claim("C06", "The survey magnetic map uses per-phone centering before node averaging/interpolation.", r"per-phone centering|phone's mean feature", [(mag_train, r"groupby\([^\n]*phone|phone.*mean|device.*mean|center")], require_all=False, notes="A semantic review is still required to ensure the grouping key is truly handset identity and not visit identity.")
    add_claim("C07", "The magnetic CNN uses Conv1D 4→32→64→128 with kernels 7,5,3 and temporal pooling 84→42→21.", r"84\\to42\\to21|84.*42.*21", [(mag_model, r"Conv1d\(4,\s*32.*kernel_size\s*=\s*7"), (mag_model, r"Conv1d\(32,\s*64.*kernel_size\s*=\s*5"), (mag_model, r"Conv1d\(64,\s*128.*kernel_size\s*=\s*3"), (mag_model, r"MaxPool1d")])
    add_claim("C08", "The magnetic CNN has a 128→64→2 position head and 128→32→1 uncertainty head.", r"128.*64.*2|128.*32.*1", [(mag_model, r"Linear\(128,\s*64\)"), (mag_model, r"Linear\(64,\s*2\)"), (mag_model, r"Linear\(128,\s*32\)"), (mag_model, r"Linear\(32,\s*1\)")])
    add_claim("C09", "Magnetic training uses 0.5||e||²/exp(ell)+0.5ell with a 0.01 floor and is not claimed as exact 2-D Gaussian NLL.", r"not.*exact negative log-likelihood|not.*exact.*Gaussian", [(mag_model, r"0\.5|/\s*var|exp\("), (mag_model, r"0\.01|clamp|min")], require_all=False)
    add_claim("C10", "PDR uses EMA alpha 0.98, threshold 0.6 m/s², 0.3 s refractory, and 0.65 m step length.", r"alpha=0\.98|tau=0\.6|L_s=0\.65", [(pdr_model, r"alpha.*0\.98"), (pdr_model, r"threshold.*0\.6|0\.6.*threshold"), (pdr_model, r"0\.3"), (fusion, r"STEP_LENGTH_M\s*=\s*0\.65")])
    add_claim("C11", "PDR bin control sums detected-step displacements.", r"sum_\{n\\in\\mathcal\{B\}_t\}", [(fusion, r"controls\[|control.*\+=|sum.*step|pdr.*control")], require_all=False)
    add_claim("C12", "Fusion uses a 13-input, 64-hidden GRUCell and an 8-output gain head reshaped into two 2×2 matrices.", r"GRUCell.*13|13-dimensional recurrent input", [(fusion, r"GRUCell\(13,\s*64\)"), (fusion, r"Linear\(64,\s*8\)"), (fusion, r"reshape|view\([^\n]*2,\s*2")])
    add_claim("C13", "The two innovations are formed against the same PDR prior.", r"dual innovation", [(fusion, r"wifi.*-.*x_pred|z_wifi.*-.*x"), (fusion, r"mag.*-.*x_pred|z_mag.*-.*x")], require_all=False)
    add_claim("C14", "Availability masks remove missing modality corrections.", r"availability mask.*zero|m_\{\\mathrm\{wifi\}", [(fusion, r"wifi_mask"), (fusion, r"mag_mask")])
    add_claim("C15", "Magnetic correction is weighted by sigmoid(-(ell-ell_ref)) = 1/(1+exp(ell-ell_ref)).", r"1\+\\exp\(\\ell_\{\\mathrm\{mag\},t\}-\\ell_\{\\mathrm\{ref\}\}\)", [(fusion, r"sigmoid\([^\n]*ref[^\n]*-.*ell|1\s*/\s*\(1\s*\+.*exp")], require_all=False)
    add_claim("C16", "The GRU receives masked innovations, Wi-Fi delta, PDR control, previous posterior displacement, masks, and clipped magnetic uncertainty.", r"Masked Wi-Fi innovation|Wi-Fi fix difference", [(fusion, r"wifi_innov"), (fusion, r"mag_innov"), (fusion, r"wifi_delta"), (fusion, r"state_delta|dx_prev|posterior"), (fusion, r"clip\([^\n]*-6[^\n]*8|clamp\([^\n]*-6[^\n]*8")], require_all=False)
    add_claim("C17", "ell_ref is the median over fusion-training magnetic scores and is frozen for evaluation.", r"median_\{\(w,t\).*D_\{\\mathrm\{train\}\}|computed only from the 250 fusion-training", [(fusion, r"median"), (fusion, r"ell_ref|logvar_ref|variance_ref")], require_all=False)
    add_claim("C18", "Fusion uses 250 training and 60 test trajectories, 160 bins, train seed 1 and test seed 2.", r"250 fusion-training trajectories and 60.*160 fusion bins", [(fusion, r"NUM_TRAIN|N_TRAIN|train.*250"), (fusion, r"NUM_TEST|N_TEST|test.*60"), (fusion, r"160"), (fusion, r"seed.*1"), (fusion, r"seed.*2")], require_all=False)
    add_claim("C19", "Exact binned trajectory signatures are checked for train/test overlap.", r"hashes the resulting binned target trajectories|identical trajectory", [(fusion, r"hashlib|sha256|signature"), (fusion, r"overlap|intersection|identical")], require_all=False)
    add_claim("C20", "Each walk is translated by its true start, so x0=0 is equivalent to a known initial position.", r"starting coordinate.*subtracted|initial 2D position is known", [(fusion, r"start.*position|positions\[0\]|target\[0\]"), (fusion, r"-\s*start|-=\s*start")], require_all=False)
    add_claim("C21", "Synthetic heading equals geometric tangent plus random-walk bias and white noise with stated scales.", r"0\.5\^\{\\circ\}/\\sqrt\{16\.7\}|8\.8\^\{\\circ\}", [(fusion, r"HEADING_DRIFT|0\.5"), (fusion, r"HEADING_NOISE|8\.8")], require_all=False)
    add_claim("C22", "Synthetic speed is uniform 1.0–1.35 m/s and gait frequency 1.7–2.0 Hz.", r"1\.0.*1\.35|1\.7--2\.0", [(fusion, r"1\.0[^\n]*1\.35"), (fusion, r"1\.7[^\n]*2\.0")])
    add_claim("C23", "Wi-Fi degraded regime uses 5 s updates and 40% AP dropout.", r"5.*s updates with 40\\% AP dropout|5.*s and independently drops 40", [(fusion, r"5\.0|WIFI_INTERVAL.*5"), (fusion, r"0\.4|40")], require_all=False)
    add_claim("C24", "Magnetic synthetic observations are map-sampled with noise and passed through causal 84-frame CNN windows.", r"bilinearly sampled.*84-frame magnetic CNN", [(fusion, r"mag.*map|interpol"), (fusion, r"84|MAG_WINDOW"), (fusion, r"noise")], require_all=False)
    add_claim("C25", "Wi-Fi MLP is trained 80 epochs, magnetic CNN 60, KalmanNet 150; optimizer/lr values match the text.", r"trained for 80 epochs.*CNN for 60.*150 epochs", [(wifi_train, r"80|epochs"), (mag_train, r"60|epochs"), (fusion, r"150|epochs"), (fusion, r"2e-3|0\.002")], require_all=False)
    add_claim("C26", "Standalone phone-split experiments hold out Samsung Galaxy S9+ from fitting.", r"Samsung Galaxy S9\+ fingerprints are held out", [(wifi_train, r"S9\+|S9"), (ROOT / "benchmarks/knn", r"S9\+|S9")], require_all=False)
    add_claim("C27", "Fusion is not presented as a held-out-device test and uses available surveyed devices in environment resources.", r"not.*held-out-smartphone|not.*unseen-device", [(fusion, r"processed.*database|all.*device|device")], require_all=False)
    add_claim("C28", "Reported headline fusion metrics are backed by machine-readable artifacts.", r"0\.494~m.*0\.473~m|1\.533~m.*1\.154~m", [(ROOT / "benchmarks/cnn_dual_kalmannet_relative_variance_metrics.json", r"0\.494|1\.154|1\.612|1\.533")], require_all=False)
    add_claim("C29", "KNN baseline metrics and held-out-device counts are backed by benchmark artifacts.", r"726 fingerprint visits.*90 S9\+|3\.31~m.*17\.54~m", [(ROOT / "benchmarks/knn", r"726|90|3\.31|17\.54|7\.46")], require_all=False)
    add_claim("C30", "Current magnetic fusion is scoped to a surveyed device-normalized domain; unseen-phone causal alignment remains future work.", r"device-normalized domain|causal alignment of an uncalibrated handset", [(mag_train, r"phone.*mean|center"), (ROOT / "paper/reviews/prof_read_ieee_comments_draft.md", r"F1\. Causal unseen-phone magnetic domain alignment")], require_all=False)

    with (OUT / "claim_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(claims[0]).keys()) if claims else ["claim_id"])
        writer.writeheader()
        for row in claims:
            writer.writerow(asdict(row))

    unsupported = [row for row in claims if row.status == "PARTIAL_OR_UNSUPPORTED"]
    if unsupported:
        add_finding(
            "MAJOR",
            "claims",
            f"{len(unsupported)} paper claims need manual code/artifact confirmation",
            ", ".join(row.claim_id for row in unsupported),
            "Inspect the evidence matrix. A regex miss is not automatically a false claim, but every listed row must be manually resolved before finalization.",
        )
    else:
        add_finding("PASS", "claims", "All scripted claim checks found supporting evidence", "All claim rows matched their configured evidence patterns.", "Still perform semantic manual review; regex support alone is not proof.")


def check_metrics() -> None:
    paper_text = read_text(ROOT / "paper/main.tex")
    metric_tokens = [
        "0.494~m", "0.437~m", "0.473~m", "0.449~m",
        "1.533~m", "1.154~m", "24.7\\%", "2.643~m", "1.612~m",
        "1.43~m", "2.02~m", "3.31~m", "17.54~m", "7.46~m",
    ]
    missing_from_paper = [token for token in metric_tokens if token not in paper_text]
    if missing_from_paper:
        add_finding("MAJOR", "metrics", "Expected headline metrics missing from manuscript", ", ".join(missing_from_paper), "Determine whether pagination/copyediting accidentally changed or removed results.")

    searchable = "\n".join(
        read_text(p)
        for p in iter_files(ROOT / "benchmarks")
        if is_text(p) and p.stat().st_size <= 2_000_000
    )
    raw_numbers = [token.replace("~m", "").replace("\\%", "") for token in metric_tokens]
    missing_from_artifacts = [num for num in raw_numbers if num not in searchable]
    if missing_from_artifacts:
        add_finding(
            "MAJOR",
            "metrics",
            "Some manuscript result values are not found in textual benchmark artifacts",
            ", ".join(missing_from_artifacts),
            "Locate the authoritative machine-readable output or regenerate it. Figures alone are insufficient provenance.",
        )
    else:
        add_finding("PASS", "metrics", "Headline numbers occur in benchmark artifacts", "All configured headline values were found under benchmarks/.", "Manual provenance review remains required to ensure each number belongs to the claimed regime/model.")


def check_stale_language() -> None:
    patterns = {
        "obsolete anomaly architecture": r"A_obs|A\(x\)|nabla A|anomaly gradient|legacy anomaly",
        "obsolete variance terminology": r"magnetic variance|log variance|relative variance",
        "stale future tense": r"next step|will replace|to be implemented|planned replacement",
        "overclaim language": r"building-agnostic|plug-and-play|entirely unseen|universally|rigorously|empirically optimal|real-time",
        "unfinished markers": r"\bTODO\b|\bFIXME\b|\bTBD\b|\bXXX\b",
        "old draft labels": r"\bV3\b|old architecture|old metrics",
    }
    active_roots = [ROOT / "README.md", ROOT / "docs", ROOT / "paper", ROOT / "benchmarks/README.md"]
    for label, pattern in patterns.items():
        hits = grep_repo(pattern, active_roots, max_hits=30)
        # Historical reviewer/legacy docs are allowed but should be explicitly marked.
        suspicious = [h for h in hits if "paper/reviews/" not in h and "legacy" not in h.lower() and "archive/" not in h]
        if suspicious:
            add_finding("MAJOR" if label in {"obsolete anomaly architecture", "overclaim language"} else "MINOR", "documentation", f"Potential {label} in active documentation", "\n".join(suspicious[:15]), "Update or remove stale language; retain historical wording only in clearly marked legacy/review records.")


def check_links() -> None:
    broken: list[str] = []
    for path in iter_files():
        if path.suffix.lower() not in {".md", ".tex"}:
            continue
        text = read_text(path)
        if path.suffix.lower() == ".md":
            for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
                target = match.group(1).strip().split("#", 1)[0]
                if not target or re.match(r"^[a-z]+://", target, re.I) or target.startswith("mailto:"):
                    continue
                candidate = (path.parent / target).resolve()
                if not candidate.exists():
                    line = text[: match.start()].count("\n") + 1
                    broken.append(f"{rel(path)}:{line} -> {target}")
        else:
            for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
                target = match.group(1)
                base = path.parent / target
                candidates = [base] if base.suffix else [base.with_suffix(ext) for ext in [".pdf", ".png", ".jpg", ".jpeg", ".eps"]]
                if not any(c.exists() for c in candidates):
                    line = text[: match.start()].count("\n") + 1
                    broken.append(f"{rel(path)}:{line} -> {target}")
    if broken:
        add_finding("MAJOR", "repository", "Broken local document/figure links", "\n".join(broken[:50]), "Repair links before deleting or moving files.")
    else:
        add_finding("PASS", "repository", "No broken local Markdown/LaTeX asset links detected", "Static path scan passed.", "None.")


def references_to(candidate: str) -> list[str]:
    escaped = re.escape(candidate)
    basename = re.escape(Path(candidate).name)
    hits = grep_repo(rf"{escaped}|{basename}", max_hits=50)
    return [h for h in hits if not h.startswith(candidate + ":")]


def cleanup_candidates(inv: dict) -> list[dict]:
    candidates: list[dict] = []

    def add(path: str, classification: str, reason: str) -> None:
        p = ROOT / path
        if not p.exists():
            return
        if p.is_dir():
            files = [f for f in p.rglob("*") if f.is_file()]
            size = sum(f.stat().st_size for f in files)
            count = len(files)
        else:
            size = p.stat().st_size
            count = 1
        refs = references_to(path)
        candidates.append({
            "path": path,
            "classification": classification,
            "reason": reason,
            "file_count": count,
            "bytes": size,
            "external_references": refs[:20],
        })

    add("agentConvoHist.md", "safe-delete", "Conversation transcript is not source, data provenance, or reproducibility documentation.")
    add("benchmarks/legacy_anomaly_fusion.py", "safe-delete", "Obsolete scalar anomaly-fusion benchmark; active paper explicitly warns not to reintroduce it.")
    add("archive/scratch", "safe-delete", "Scratch workspace is superseded and should not live in the final research repository.")
    add("archive/legacy_publication", "safe-delete", "Superseded paper drafts and publication sandboxes are preserved by Git history.")
    add("archive/legacy_experiments", "safe-delete", "Superseded experimental code is preserved by Git history; keeping it in-tree risks accidental use.")
    add("archive/dataset_generated", "review-delete", "Large derived/generated legacy datasets duplicate reconstructable artifacts and dominate repository size; verify no active script consumes them.")
    add("docs/project_history", "review-delete", "Historical reports can be recovered from Git history and may contain stale claims.")
    add("docs/architecture/kalman_fusion_legacy.md", "safe-delete", "Explicitly legacy architecture document.")
    add("paper/notes", "review-delete", "Internal audit notes may be stale; retain only the new final evidence audit and any still-authoritative provenance notes.")
    add("references", "review-delete", "Bundled paper/reference files are not needed if BibTeX metadata and stable citations suffice; verify licensing and offline reproducibility needs.")

    (OUT / "cleanup_candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    return candidates


def snapshot() -> None:
    snapshot_root = OUT / "active_snapshot"
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    snapshot_root.mkdir()
    include = ["README.md", ".gitignore", ".gitattributes", "requirements.txt", "models", "train", "benchmarks", "data", "docs", "paper", "tools"]
    for item in include:
        src = ROOT / item
        if not src.exists():
            continue
        dst = snapshot_root / item
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build", "audit_output", "tmp_pre_final_audit.py"),
            )
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    # Avoid shipping large raw/processed data in the audit artifact; retain manifests only.
    for path in list((snapshot_root / "data").rglob("*")) if (snapshot_root / "data").exists() else []:
        if path.is_file() and path.stat().st_size > 2_000_000:
            path.unlink()
    shutil.make_archive(str(OUT / "active_source_snapshot"), "zip", snapshot_root)
    shutil.rmtree(snapshot_root)


def render_report(inv: dict, cleanup: list[dict]) -> None:
    severity_order = {"BLOCKER": 0, "MAJOR": 1, "MINOR": 2, "PASS": 3, "INFO": 4}
    ordered = sorted(findings, key=lambda f: (severity_order.get(f.severity, 9), f.category, f.title))
    lines: list[str] = []
    lines.append("# Pre-final Paper, Code, Claims, and Repository Audit")
    lines.append("")
    lines.append("This report is generated from the exact checked-out commit. Scripted matches are evidence aids, not substitutes for semantic review. A `PARTIAL_OR_UNSUPPORTED` row may be a regex miss; it must nevertheless be resolved manually before the manuscript is called final.")
    lines.append("")
    lines.append("## Executive counts")
    lines.append("")
    counts = Counter(f.severity for f in ordered)
    lines.append(f"- Findings: {dict(counts)}")
    claim_counts = Counter(c.status for c in claims)
    lines.append(f"- Claim checks: {dict(claim_counts)}")
    lines.append(f"- Repository files: {inv['file_count']:,}")
    lines.append(f"- Repository working-tree bytes: {inv['total_bytes']:,}")
    lines.append(f"- Cleanup candidates: {len(cleanup)}")
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    for f in ordered:
        lines.append(f"### [{f.severity}] {f.title}")
        lines.append("")
        lines.append(f"**Category:** {f.category}")
        lines.append("")
        lines.append("**Evidence**")
        lines.append("")
        lines.append("```text")
        lines.append(f.evidence[:12000])
        lines.append("```")
        lines.append("")
        lines.append(f"**Recommendation:** {f.recommendation}")
        lines.append("")

    lines.append("## Claim-to-code evidence matrix")
    lines.append("")
    for row in claims:
        lines.append(f"### {row.claim_id} — {row.status}")
        lines.append("")
        lines.append(f"**Claim:** {row.paper_claim}")
        lines.append("")
        lines.append("**Paper location**")
        lines.append("")
        lines.append("```text")
        lines.append(row.paper_location)
        lines.append("```")
        lines.append("")
        lines.append("**Evidence**")
        lines.append("")
        lines.append("```text")
        lines.append(row.evidence[:12000])
        lines.append("```")
        if row.notes:
            lines.append("")
            lines.append(f"**Notes:** {row.notes}")
        lines.append("")

    lines.append("## Repository inventory")
    lines.append("")
    lines.append("### Top-level file counts and bytes")
    lines.append("")
    lines.append("| Path | Files | Bytes |")
    lines.append("|---|---:|---:|")
    for top, count in inv["top_level_counts"].items():
        lines.append(f"| `{top}` | {count:,} | {inv['top_level_bytes'].get(top, 0):,} |")
    lines.append("")
    lines.append("### Largest files")
    lines.append("")
    for item in inv["large_files"][:50]:
        lines.append(f"- `{item['path']}` — {item['bytes']:,} bytes")
    lines.append("")
    lines.append("### Duplicate groups (first 30)")
    lines.append("")
    for group in inv["duplicate_groups"][:30]:
        lines.append("- " + ", ".join(f"`{p}`" for p in group))
    lines.append("")

    lines.append("## Cleanup candidates")
    lines.append("")
    lines.append("No candidate should be deleted until its external-reference list and reproducibility role are reviewed.")
    lines.append("")
    for item in cleanup:
        lines.append(f"### {item['classification']}: `{item['path']}`")
        lines.append("")
        lines.append(f"- Reason: {item['reason']}")
        lines.append(f"- Files: {item['file_count']:,}")
        lines.append(f"- Bytes: {item['bytes']:,}")
        if item["external_references"]:
            lines.append("- External references:")
            for ref_hit in item["external_references"]:
                lines.append(f"  - `{ref_hit}`")
        else:
            lines.append("- External references: none found by static text scan")
        lines.append("")

    (OUT / "pre_final_audit.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "findings.json").write_text(json.dumps([asdict(f) for f in ordered], indent=2), encoding="utf-8")


def main() -> int:
    inv = inventory()
    check_python()
    smoke_models()
    check_claims()
    check_metrics()
    check_stale_language()
    check_links()
    cleanup = cleanup_candidates(inv)
    snapshot()
    render_report(inv, cleanup)

    print((OUT / "pre_final_audit.md").read_text(encoding="utf-8")[:20000])
    blockers = [f for f in findings if f.severity == "BLOCKER"]
    print(f"\nAudit completed with {len(blockers)} blocker(s), {len(findings)} total finding(s), and {len(claims)} claim rows.")
    # Do not fail on claim regex misses; fail only on executable blockers.
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
