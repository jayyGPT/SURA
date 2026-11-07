#!/usr/bin/env python3
"""Reproducible streaming fusion-stage latency benchmark.

Uses prepared PDR controls, Wi-Fi/magnetic fixes, confidence scores and masks.
Measurement networks, preprocessing, loading and training are outside the timer.
Default: CPU, one intra-op/inter-op thread, eight passes with balanced block order.
Passes reuse the same models and data in one process; they are not independent
machine runs. All observations, including scheduling pauses, are retained.

    python benchmarks/latency_benchmark.py
    python benchmarks/latency_benchmark.py --cpu-affinity 8 --output runs/cpu.csv
    python benchmarks/latency_benchmark.py --print-schedule
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
import time

METHOD_ORDER = ['wifi_only_kalmannet', 'cnn_dual_kalmannet', 'knn_k5', 'knn_k20']


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return result


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return result


def make_schedule(regimes, repeats, policy='balanced', seed=1729):
    """Williams design: each condition occupies each position once per cycle.

    With four/eight conditions, a complete cycle also balances immediate
    predecessor pairs within passes. Partial cycles remain explicitly recorded.
    """
    tasks = [(regime, method) for regime in regimes for method in METHOD_ORDER]
    n = len(tasks)
    rng = random.Random(seed)
    labels = tasks.copy()
    if policy in ('balanced', 'random'):
        rng.shuffle(labels)
    first_row = [0] + [(i + 1) // 2 if i % 2 else n - i // 2 for i in range(1, n)]
    result = []
    for repeat in range(1, repeats + 1):
        if policy == 'balanced':
            row = [labels[(index + repeat - 1) % n] for index in first_row]
        elif policy == 'random':
            row = tasks.copy()
            rng.shuffle(row)
        else:
            ordered_regimes = ['degraded', 'full'] if policy == 'degraded_first' else ['full', 'degraded']
            row = [(regime, method) for regime in ordered_regimes if regime in regimes for method in METHOD_ORDER]
        for position, (regime, method) in enumerate(row, 1):
            result.append({'repeat': repeat, 'position_in_repeat': position,
                           'regime': regime, 'method': method, 'block_index': len(result) + 1})
    return result


def configure_process(threads, affinity):
    """Set library environment before importing numerical runtimes."""
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = str(threads)
    before = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    if affinity != 'none' and hasattr(os, 'sched_setaffinity'):
        if affinity == 'auto':
            selected = before[:threads]
        else:
            selected = []
            for part in affinity.split(','):
                if '-' in part:
                    start, stop = map(int, part.split('-', 1))
                    if start > stop:
                        raise ValueError('CPU affinity range must be ascending')
                    selected.extend(range(start, stop + 1))
                else:
                    selected.append(int(part))
            selected = sorted(set(selected))
        if not selected or not set(selected).issubset(before):
            raise ValueError(f'CPU selection must be inside allowed CPUs: {before}')
        if len(selected) < threads:
            raise ValueError('CPU affinity contains fewer CPUs than requested intra-op threads')
        os.sched_setaffinity(0, selected)
        after = sorted(os.sched_getaffinity(0))
    elif affinity not in ('auto', 'none'):
        raise RuntimeError('Explicit CPU affinity is supported on Linux only; use --cpu-affinity none here')
    else:
        after = before
    return {'requested': affinity, 'allowed_before': before, 'effective': after,
            'exclusive_reservation': False,
            'note': 'auto selects the first allowed CPUs; it does not detect idle CPUs or reserve a core'}


def command_output(command, cwd=None):
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                                timeout=10, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def read_text(path):
    try:
        return Path(path).read_text(encoding='utf-8').strip()
    except (OSError, UnicodeError):
        return None


def runtime_snapshot():
    result = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'process_cpu_seconds': time.process_time()}
    if hasattr(os, 'getloadavg'):
        result['load_average_1_5_15_min'] = list(os.getloadavg())
    if sys.platform.startswith('linux'):
        cpus = sorted(os.sched_getaffinity(0))
        result['selected_cpu_status'] = {
            str(cpu): {'frequency_khz': read_text(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq'),
                       'governor': read_text(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor')}
            for cpu in cpus
        }
        status = read_text('/proc/self/status') or ''
        result['process_status'] = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                                    for line in status.splitlines() if line.startswith(
                                        ('VmRSS:', 'voluntary_ctxt_switches:', 'nonvoluntary_ctxt_switches:'))}
    return result


def collect_system_metadata(args, device, binding):
    cpu = {'logical_cores': os.cpu_count(), 'name': platform.processor()}
    power = battery = None
    if sys.platform.startswith('linux'):
        raw = command_output(['lscpu', '-J'])
        if raw:
            try:
                fields = {row['field'].rstrip(':'): row['data'] for row in json.loads(raw)['lscpu']}
                cpu['lscpu'] = fields
                cpu['name'] = fields.get('Model name', cpu['name'])
                if 'Socket(s)' in fields and 'Core(s) per socket' in fields:
                    cpu['physical_cores'] = int(fields['Socket(s)']) * int(fields['Core(s) per socket'])
            except (ValueError, KeyError, TypeError):
                cpu['lscpu_raw'] = raw
        cpu['topology_csv'] = command_output(['lscpu', '-p=CPU,CORE,SOCKET,NODE'])
        power = command_output(['powerprofilesctl', 'get'])
        battery = {entry.name: {'type': read_text(entry / 'type'),
                               'status': read_text(entry / 'status'),
                               'online': read_text(entry / 'online')}
                   for entry in Path('/sys/class/power_supply').glob('*')}
    elif os.name == 'nt':
        raw = command_output(['powershell', '-NoProfile', '-Command',
                              'Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | ConvertTo-Json'])
        if raw:
            try:
                chips = json.loads(raw)
                chips = chips if isinstance(chips, list) else [chips]
                cpu.update(name='; '.join(chip['Name'].strip() for chip in chips),
                           physical_cores=sum(chip['NumberOfCores'] for chip in chips),
                           logical_cores=sum(chip['NumberOfLogicalProcessors'] for chip in chips))
            except (ValueError, KeyError, TypeError):
                cpu['windows_cpu_raw'] = raw
        power = command_output(['powercfg', '/getactivescheme'])
        battery = command_output(['powershell', '-NoProfile', '-Command',
                                  'Get-CimInstance Win32_Battery | Select-Object BatteryStatus | ConvertTo-Json'])
    clock = time.get_clock_info('perf_counter')
    return {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'hostname': socket.gethostname(),
            'platform': platform.platform(), 'python_version': sys.version,
            'torch_version': torch.__version__, 'sklearn_version': sklearn.__version__,
            'numpy_version': np.__version__, 'scipy_version': importlib.metadata.version('scipy'),
            'threadpoolctl_version': importlib.metadata.version('threadpoolctl'),
            'device': str(device), 'cpu': cpu, 'power_scheme': power, 'battery_status': battery,
            'cpu_affinity': binding, 'torch_build_configuration': torch.__config__.show(),
            'thread_settings': {'torch_num_threads': torch.get_num_threads(),
                                'torch_interop_threads': torch.get_num_interop_threads(),
                                'library_environment': {name: os.environ.get(name) for name in
                                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS')},
                                'effective_threadpools': threadpool_info()},
            'timer': {'implementation': clock.implementation, 'resolution_seconds': clock.resolution,
                      'monotonic': clock.monotonic,
                      'cuda_synchronization': device.type == 'cuda'},
            'start_snapshot': runtime_snapshot()}


def load_runtime(repo_root):
    """Import scientific libraries only after process settings are applied."""
    global np, scipy, sklearn, torch, KNeighborsRegressor, threadpool_info, threadpool_limits
    global WiFiOnlyKalmanNet, CNNMagneticDualKalmanNet, setup_environment, make_dataset, prepare_tensors
    global DEFAULT_DATABASE, DEFAULT_WIFI_CHECKPOINT, DEFAULT_MAG_CHECKPOINT, DEFAULT_OUTPUT
    global FUSION_HIDDEN_SIZE, FUSION_TRAIN_SEED, FUSION_FINAL_TEST_SEED
    import numpy as np
    import scipy.stats
    import sklearn
    from sklearn.neighbors import KNeighborsRegressor
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    sys.path.insert(0, str(repo_root))
    from models.wifi_kalmannet import WiFiOnlyKalmanNet
    from train.kalmannet_wifiheatmap_magneticCNN_pdr import (
        CNNMagneticDualKalmanNet, DEFAULT_DATABASE, DEFAULT_WIFI_CHECKPOINT,
        DEFAULT_MAG_CHECKPOINT, DEFAULT_OUTPUT, FUSION_HIDDEN_SIZE,
        FUSION_TRAIN_SEED, FUSION_FINAL_TEST_SEED, setup_environment, make_dataset, prepare_tensors)


def benchmark_wifi_kalmannet(model: WiFiOnlyKalmanNet, tensors: tuple, walk_idx: int, *, capture_predictions: bool = False):
    """Time each GRU step for one walk. Returns list of per-step ms."""
    motion, wifi, wifi_mask = tensors[0], tensors[1], tensors[2]
    m = motion[walk_idx : walk_idx + 1]
    w = wifi[walk_idx : walk_idx + 1]
    wm = wifi_mask[walk_idx : walk_idx + 1]

    steps = m.shape[1]
    hidden = m.new_zeros(1, model.hidden_size)
    state = m.new_zeros(1, 2)
    previous_wifi = w[:, 0]
    previous_update = m.new_zeros(1, 2)
    timings: list[float] = []
    outputs = [] if capture_predictions else None
    is_cuda = m.device.type == "cuda"

    for step in range(steps):
        if is_cuda:
            torch.cuda.synchronize(m.device)
        t0 = time.perf_counter()
        mask = wm[:, step]
        predicted = state + m[:, step]
        innovation = (w[:, step] - predicted) * mask
        wifi_delta = (w[:, step] - previous_wifi) * mask
        features = torch.cat(
            [innovation, wifi_delta, m[:, step], previous_update, mask], dim=1
        )
        hidden = model.cell(features, hidden)
        gain = model.head(hidden).view(1, 2, 2)
        correction = torch.bmm(gain, innovation.unsqueeze(-1)).squeeze(-1) * mask
        updated = predicted + correction
        previous_update = updated - state
        previous_wifi = torch.where(mask.bool(), w[:, step], previous_wifi)
        state = updated
        if is_cuda:
            torch.cuda.synchronize(m.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed_ms)
        if outputs is not None:
            outputs.append(state.clone())

    if outputs is not None:
        return timings, torch.stack(outputs, dim=1)
    return timings


def benchmark_dual_kalmannet(model: CNNMagneticDualKalmanNet, tensors: tuple, walk_idx: int, *, capture_predictions: bool = False):
    """Time each GRU step for one walk. Returns list of per-step ms."""
    motion, wifi, wifi_mask, mag, mag_logvar, mag_mask = tensors[:6]
    m = motion[walk_idx : walk_idx + 1]
    w = wifi[walk_idx : walk_idx + 1]
    wm = wifi_mask[walk_idx : walk_idx + 1]
    mg = mag[walk_idx : walk_idx + 1]
    mlv = mag_logvar[walk_idx : walk_idx + 1]
    mm = mag_mask[walk_idx : walk_idx + 1]

    steps = m.shape[1]
    hidden = m.new_zeros(1, model.hidden_size)
    state = m.new_zeros(1, 2)
    previous_wifi = w[:, 0]
    previous_update = m.new_zeros(1, 2)
    timings: list[float] = []
    outputs = [] if capture_predictions else None
    is_cuda = m.device.type == "cuda"

    for step in range(steps):
        if is_cuda:
            torch.cuda.synchronize(m.device)
        t0 = time.perf_counter()
        wifi_available = wm[:, step]
        mag_available = mm[:, step]
        predicted = state + m[:, step]
        wifi_innovation = (w[:, step] - predicted) * wifi_available
        magnetic_innovation = (mg[:, step] - predicted) * mag_available
        wifi_delta = (w[:, step] - previous_wifi) * wifi_available
        mag_confidence = mlv[:, step].clamp(-6.0, 8.0) * mag_available
        relative_lv = (
            mlv[:, step] - model.magnetic_reference_log_variance
        ).clamp(-8.0, 8.0)
        magnetic_weight = 1.0 / (1.0 + torch.exp(relative_lv))
        features = torch.cat(
            [
                wifi_innovation,
                magnetic_innovation,
                wifi_delta,
                m[:, step],
                previous_update,
                wifi_available,
                mag_available,
                mag_confidence,
            ],
            dim=1,
        )
        hidden = model.cell(features, hidden)
        gains = model.head(hidden)
        wifi_gain = gains[:, :4].view(1, 2, 2)
        magnetic_gain = gains[:, 4:].view(1, 2, 2)
        wifi_correction = wifi_available * torch.bmm(
            wifi_gain, wifi_innovation.unsqueeze(-1)
        ).squeeze(-1)
        magnetic_correction = magnetic_weight * mag_available * torch.bmm(
            magnetic_gain, magnetic_innovation.unsqueeze(-1)
        ).squeeze(-1)
        updated = predicted + wifi_correction + magnetic_correction
        previous_update = updated - state
        previous_wifi = torch.where(
            wifi_available.bool(), w[:, step], previous_wifi
        )
        state = updated
        if is_cuda:
            torch.cuda.synchronize(m.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed_ms)
        if outputs is not None:
            outputs.append(state.clone())

    if outputs is not None:
        return timings, torch.stack(outputs, dim=1)
    return timings


def trajectory_features(data, reference_logvar: float):
    """Extract per-step features for KNN (same logic as wifi_mag_knn.py)."""
    _, wifi, wifi_mask, mag, mag_logvar, mag_mask, target, _ = data
    mag_available = mag_mask[..., 0] > 0.5
    wifi_available = wifi_mask[..., 0] > 0.5
    mag_filled = mag.copy()
    mag_filled[~mag_available] = wifi[~mag_available]
    logvar = np.clip(mag_logvar[..., 0], -6.0, 8.0)
    logvar[~mag_available] = reference_logvar
    features = np.concatenate(
        (
            wifi,
            mag_filled,
            logvar[..., None],
            wifi_available[..., None].astype(np.float32),
            mag_available[..., None].astype(np.float32),
        ),
        axis=2,
    )
    walks, bins, channels = features.shape
    return features.reshape(walks * bins, channels), target.reshape(walks * bins, 2)



def validate_neural_steps(model, tensors, dual):
    """Verify the timed update reproduces the ordinary forward pass."""
    with torch.no_grad():
        inputs = tensors[:6] if dual else tensors[:3]
        expected = model(*(value[:1] for value in inputs))
        helper = benchmark_dual_kalmannet if dual else benchmark_wifi_kalmannet
        _, actual = helper(model, tensors, 0, capture_predictions=True)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    return {'passed': True, 'max_abs_difference': float((actual - expected).abs().max().item()),
            'walks_checked': 1, 'bins_checked': int(actual.shape[1])}


def prepare_regime(args, regime, env, device):
    period, dropout = (1.0, 0.0) if regime == 'full' else (5.0, 0.4)
    print(f'Preparing {regime} data and models (outside timing)...', flush=True)
    training = make_dataset(args.train_walks, seed=FUSION_TRAIN_SEED, env=env, device=device,
                            wifi_period_s=period, ap_dropout=dropout, bins=args.bins)
    testing = make_dataset(args.test_walks, seed=FUSION_FINAL_TEST_SEED, env=env, device=device,
                           wifi_period_s=period, ap_dropout=dropout, bins=args.bins)
    folder = args.kalmannet_dir / regime
    wifi_path = folder / 'wifi_only_kalmannet.pt'
    dual_path = folder / 'cnn_dual_relative_confidence.pt'
    baseline = WiFiOnlyKalmanNet(hidden_size=FUSION_HIDDEN_SIZE).to(device)
    baseline.load_state_dict(torch.load(wifi_path, map_location=device, weights_only=False)['state_dict'])
    baseline.eval()
    payload = torch.load(dual_path, map_location=device, weights_only=False)
    dual = CNNMagneticDualKalmanNet(hidden_size=FUSION_HIDDEN_SIZE,
        magnetic_reference_log_variance=payload.get('magnetic_reference_log_variance', 0.0)).to(device)
    dual.load_state_dict(payload['state_dict'])
    dual.eval()
    tensors = prepare_tensors(testing, device)
    available = training[5][..., 0] > 0.5
    reference = float(np.median(training[4][..., 0][available])) if available.any() else 0.0
    x_train, y_train = trajectory_features(training, reference)
    x_test, _ = trajectory_features(testing, reference)
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_scaled = ((x_train - mean) / std).astype(np.float32)
    test_scaled = ((x_test - mean) / std).astype(np.float32)
    models = {'wifi_only_kalmannet': baseline, 'cnn_dual_kalmannet': dual}
    knn_metadata = {}
    for k in (5, 20):
        knn = KNeighborsRegressor(n_neighbors=k, metric='euclidean', weights='uniform', algorithm='auto', n_jobs=1)
        knn.fit(train_scaled, y_train)
        models[f'knn_k{k}'] = knn
        knn_metadata[f'knn_k{k}'] = {'parameters': knn.get_params(),
            'reference_points': int(knn.n_samples_fit_), 'features': int(knn.n_features_in_),
            'effective_search_algorithm': getattr(knn, '_fit_method', 'unavailable')}
    dataset_hashes = {}
    for label, data in [('training', training), ('testing', testing)]:
        dataset_hashes[label] = [hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for a in data]
    details = {'wifi_period_s': period, 'ap_dropout': dropout,
        'training_walks': len(training[0]), 'test_walks': len(testing[0]), 'bins_per_walk': args.bins,
        'wifi_available_steps': int(testing[2].sum()), 'magnetic_available_steps': int(testing[5].sum()),
        'knn': knn_metadata, 'dataset_array_sha256': dataset_hashes,
        'checkpoints': {'wifi_only': file_identity(wifi_path), 'dual': file_identity(dual_path)},
        'parameter_counts': {'wifi_only': sum(p.numel() for p in baseline.parameters()),
                             'dual': sum(p.numel() for p in dual.parameters())},
        'forward_equivalence': {'wifi_only': validate_neural_steps(baseline, tensors, False),
                                'dual': validate_neural_steps(dual, tensors, True)}}
    return {'models': models, 'tensors': tensors, 'x_test': test_scaled,
            'wifi_mask': testing[2][..., 0] > 0.5, 'mag_mask': testing[5][..., 0] > 0.5}, details


def warmup_block(resource, method, args):
    if method.startswith('knn_'):
        for i in range(min(args.warmup_queries, len(resource['x_test']))):
            resource['models'][method].predict(resource['x_test'][i:i+1])
    else:
        helper = benchmark_dual_kalmannet if method == 'cnn_dual_kalmannet' else benchmark_wifi_kalmannet
        with torch.no_grad():
            for walk in range(min(args.warmup_walks, args.test_walks)):
                helper(resource['models'][method], resource['tensors'], walk)


def summarize(rows, regimes, repeats, cv_warning):
    results = {}
    warnings = []
    for regime in regimes:
        results[regime] = {}
        for method in METHOD_ORDER:
            selected = [row for row in rows if row['regime'] == regime and row['method'] == method]
            means, medians, repeat_details = [], [], []
            for repeat in range(1, repeats + 1):
                values = np.array([row['latency_ms'] for row in selected if row['repeat'] == repeat])
                means.append(float(values.mean()))
                medians.append(float(np.median(values)))
                repeat_details.append({'repeat': repeat, 'count': len(values), 'mean_ms': means[-1],
                    'median_ms': medians[-1], 'p95_ms': float(np.percentile(values, 95)),
                    'p99_ms': float(np.percentile(values, 99)), 'max_ms': float(values.max())})
            pooled = np.array([row['latency_ms'] for row in selected])
            mean = float(np.mean(means))
            sd = float(np.std(means, ddof=1)) if repeats > 1 else None
            cv = 100.0 * sd / mean if sd is not None else None
            ci = float(scipy.stats.t.ppf(.975, repeats - 1) * sd / math.sqrt(repeats)) if sd is not None else None
            flag = cv is not None and cv > cv_warning
            results[regime][method] = {'repeat_means_ms': means, 'repeat_medians_ms': medians,
                'repeat_details': repeat_details, 'inter_repeat_mean_ms': mean,
                'inter_repeat_std_ms': sd, 'coefficient_of_variation_pct': cv, 'ci95_ms': ci,
                'ci95_interpretation': 'Student-t half-width across same-process pass means; assumes independent, approximately normal pass means',
                'pooled_mean_ms': float(pooled.mean()), 'pooled_median_ms': float(np.median(pooled)),
                'pooled_std_ms': float(pooled.std()), 'pooled_p95_ms': float(np.percentile(pooled, 95)),
                'pooled_p99_ms': float(np.percentile(pooled, 99)), 'pooled_max_ms': float(pooled.max()),
                'pooled_min_ms': float(pooled.min()), 'count': len(pooled), 'repeat_variation_flag': flag}
            cv_label = f'{cv:.2f}%' if cv is not None else 'unavailable (one pass)'
            print(f'{regime:8} {method:24} mean={mean:.5f} ms; pass CV={cv_label}; P95={np.percentile(pooled, 95):.5f} ms')
            if flag:
                warnings.append(f'{regime}/{method}: pass-mean CV {cv:.2f}% exceeds descriptive threshold {cv_warning:g}%')
    return results, warnings


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    for name in ('database', 'wifi-checkpoint', 'mag-checkpoint', 'kalmannet-dir', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--regime', choices=['both', 'full', 'degraded'], default='both')
    parser.add_argument('--order', choices=['balanced', 'random', 'full_first', 'degraded_first'], default='balanced')
    parser.add_argument('--order-seed', type=int, default=1729)
    parser.add_argument('--repeats', type=positive_int, default=8)
    parser.add_argument('--threads', type=positive_int, default=1)
    parser.add_argument('--interop-threads', type=positive_int, default=1)
    parser.add_argument('--cpu-affinity', default='auto', help='Linux: auto selects first allowed CPUs; none disables; or give a list such as 8 or 8-15')
    parser.add_argument('--train-walks', type=positive_int, default=250)
    parser.add_argument('--test-walks', type=positive_int, default=60)
    parser.add_argument('--bins', type=positive_int, default=160)
    parser.add_argument('--warmup-walks', type=nonnegative_int, default=3)
    parser.add_argument('--warmup-queries', type=nonnegative_int, default=200)
    parser.add_argument('--cv-warning-percent', type=float, default=5.0)
    parser.add_argument('--print-schedule', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.cv_warning_percent) or args.cv_warning_percent < 0:
        raise ValueError('--cv-warning-percent must be finite and nonnegative')
    if args.train_walks * args.bins < 20:
        raise ValueError('K=20 requires at least 20 training bins')
    regimes = ['full', 'degraded'] if args.regime == 'both' else [args.regime]
    schedule = make_schedule(regimes, args.repeats, args.order, args.order_seed)
    if args.print_schedule:
        print(json.dumps(schedule, indent=2))
        return 0
    root = args.repo_root.resolve()
    binding = configure_process(args.threads, args.cpu_affinity)
    load_runtime(root)
    torch.set_num_threads(args.threads)
    if torch.get_num_interop_threads() != args.interop_threads:
        torch.set_num_interop_threads(args.interop_threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    for name, default in [('database', DEFAULT_DATABASE), ('wifi_checkpoint', DEFAULT_WIFI_CHECKPOINT),
                          ('mag_checkpoint', DEFAULT_MAG_CHECKPOINT), ('kalmannet_dir', DEFAULT_OUTPUT)]:
        setattr(args, name, (getattr(args, name) or default).resolve())
    if args.output is None:
        args.output = root / 'benchmarks' / 'runs' / ('latency_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.csv')
    args.output = args.output.resolve()
    meta_path = args.output.with_suffix('.metadata.json')
    if not args.overwrite and (args.output.exists() or meta_path.exists()):
        raise FileExistsError('Output already exists; choose a new --output or explicitly pass --overwrite')
    checkpoint_paths = [args.wifi_checkpoint, args.mag_checkpoint]
    for regime in regimes:
        checkpoint_paths.extend([args.kalmannet_dir / regime / 'wifi_only_kalmannet.pt',
                                 args.kalmannet_dir / regime / 'cnn_dual_relative_confidence.pt'])
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(f'Required checkpoint missing: {path}')
    rows = []
    run_start = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        metadata = collect_system_metadata(args, device, binding)
        metadata.update(status='running', arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            invocation=[sys.executable, *sys.argv], repository={'root': str(root),
                'git_commit': command_output(['git', 'rev-parse', 'HEAD'], root),
                'git_status_porcelain': command_output(['git', 'status', '--porcelain'], root)},
            source_files=[file_identity(path) for path in [Path(__file__), root / 'models/wifi_kalmannet.py',
                          root / 'train/kalmannet_wifiheatmap_magneticCNN_pdr.py']],
            measurement_checkpoints=[file_identity(args.wifi_checkpoint), file_identity(args.mag_checkpoint)],
            benchmark_protocol={'scope': 'fusion/prediction with prepared measurement inputs; no measurement-network inference, preprocessing, loading or training',
                'batch_size_or_queries_per_call': 1, 'repeats': args.repeats,
                'repeat_design': 'same-process passes over identical data; no automatic independence claim',
                'training_seed': FUSION_TRAIN_SEED, 'test_seed': FUSION_FINAL_TEST_SEED,
                'warmup_walks_per_neural_block': args.warmup_walks, 'warmup_queries_per_knn_block': args.warmup_queries,
                'order_policy': args.order, 'order_seed': args.order_seed, 'schedule': schedule,
                'complete_position_balance': args.order == 'balanced' and args.repeats % (len(regimes) * len(METHOD_ORDER)) == 0,
                'cv_warning_percent': args.cv_warning_percent,
                'outlier_policy': 'retain all observations; warnings are diagnostic, not quality certification'},
            regimes={}, blocks=[])
        try:
            env = setup_environment(args.database, args.wifi_checkpoint, args.mag_checkpoint, device)
            resources = {}
            # Prepare both regimes before the first timing block.
            for regime in regimes:
                resources[regime], metadata['regimes'][regime] = prepare_regime(args, regime, env, device)
            for block in schedule:
                regime, method = block['regime'], block['method']
                resource = resources[regime]
                print(f"Block {block['block_index']}/{len(schedule)}: pass {block['repeat']}, {regime}, {method}", flush=True)
                warmup_block(resource, method, args)
                block_record = dict(block, start_snapshot=runtime_snapshot(),
                                    started_elapsed_s=time.perf_counter() - run_start)
                for walk in range(args.test_walks):
                    if method.startswith('knn_'):
                        timings = []
                        for step in range(args.bins):
                            index = walk * args.bins + step
                            query = resource['x_test'][index:index+1]
                            t0 = time.perf_counter()
                            resource['models'][method].predict(query)
                            elapsed_ms = (time.perf_counter() - t0) * 1000.0
                            timings.append(elapsed_ms)
                    else:
                        helper = benchmark_dual_kalmannet if method == 'cnn_dual_kalmannet' else benchmark_wifi_kalmannet
                        with torch.no_grad():
                            timings = helper(resource['models'][method], resource['tensors'], walk)
                    if len(timings) != args.bins or any(not math.isfinite(v) or v <= 0 for v in timings):
                        raise RuntimeError('Invalid or incomplete timing block')
                    rows.extend({'repeat': block['repeat'], 'method': method, 'regime': regime,
                        'walk': walk, 'step': step, 'latency_ms': ms, 'block_index': block['block_index'],
                        'wifi_available': int(resource['wifi_mask'][walk, step]),
                        'mag_available': int(resource['mag_mask'][walk, step])} for step, ms in enumerate(timings))
                block_record.update(end_snapshot=runtime_snapshot(), finished_elapsed_s=time.perf_counter() - run_start,
                                    count=args.test_walks * args.bins)
                metadata['blocks'].append(block_record)
            expected = args.repeats * len(regimes) * len(METHOD_ORDER) * args.test_walks * args.bins
            if len(rows) != expected:
                raise RuntimeError(f'Incomplete run: {len(rows)} rows instead of {expected}')
            metadata['results'], metadata['warnings'] = summarize(rows, regimes, args.repeats, args.cv_warning_percent)
            metadata['status'] = 'complete'
            metadata['expected_rows'] = expected
        except Exception as error:
            metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            metadata.update(recorded_rows=len(rows), end_snapshot=runtime_snapshot(),
                            finished_utc=datetime.now(timezone.utc).isoformat(),
                            elapsed_seconds=time.perf_counter() - run_start)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            fields = ['repeat', 'method', 'regime', 'walk', 'step', 'latency_ms', 'block_index', 'wifi_available', 'mag_available']
            with args.output.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            metadata['csv_sha256'] = sha256_file(args.output)
            meta_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    print(f'CSV: {args.output}\nMetadata: {meta_path}')
    for warning in metadata['warnings']:
        print('VARIATION WARNING: ' + warning)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
