import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

import cv2

from .classifier import (
    annotate,
    detect_proposals,
    detection_metrics,
    load_config,
    select_winner,
)
from .profiles import (
    CAMERA_PROFILES,
    DEFAULT_CAMERA_PROFILE,
    dataset_path,
    detector_config_path,
    require_calibrated_detector,
    results_path,
)


CLASSES = ('BLUE', 'CYAN', 'GREEN', 'PURPLE', 'RED', 'NONE')


def _different_color_margin(candidates) -> float:
    if not candidates:
        return math.inf
    different = next(
        (item for item in candidates[1:]
         if item.color != candidates[0].color),
        None)
    if different is None:
        return math.inf
    return candidates[0].score / max(different.score, 1e-9)


def _evaluation_passed(
        failures: int,
        separation_passed: bool,
        p95_ms: float,
        none_validation: str) -> bool:
    """只有经过有效负样本验证的检测器才能通过验收。"""
    return (
        failures == 0 and separation_passed and p95_ms < 60.0 and
        none_validation == 'passed'
    )


def _detect_with_scale(image, config, processing_scale: float):
    work = image
    if processing_scale < 1.0:
        work = cv2.resize(
            image, None, fx=processing_scale, fy=processing_scale,
            interpolation=cv2.INTER_AREA)
    proposals = detect_proposals(work, config)
    if processing_scale < 1.0:
        inverse = 1.0 / processing_scale
        proposals = [item.scaled(inverse) for item in proposals]
    return proposals


def evaluate(
        dataset: Path,
        output: Path,
        config_path: Path,
        processing_scale: float = 1.0) -> int:
    config = load_config(str(config_path))
    processing_scale = max(0.1, min(float(processing_scale), 1.0))
    records: List[Dict] = []
    durations = []
    class_counts = {name: len(list((dataset / name).glob('*.jpg')))
                    for name in CLASSES}
    for expected in CLASSES:
        for image_path in sorted((dataset / expected).glob('*.jpg')):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f'cannot read {image_path}')
            started = time.perf_counter()
            proposals = _detect_with_scale(image, config, processing_scale)
            duration = time.perf_counter() - started
            durations.append(duration)
            candidates = [
                item for item in proposals if item.score >= config.min_score
            ]
            winner = select_winner(candidates, config)
            records.append({
                'path': image_path,
                'expected': expected,
                'proposals': proposals,
                'candidates': candidates,
                'winner': winner,
                'margin': _different_color_margin(candidates),
                'processing_ms': duration * 1000.0,
            })

    true_scores = [
        row['winner'].score for row in records
        if row['expected'] != 'NONE' and row['winner'] is not None and
        row['winner'].color == row['expected']
    ]
    false_scores = [
        item.score
        for row in records
        for item in row['proposals']
        if (
            row['expected'] == 'NONE' or
            item.color != row['expected']
        )
    ]
    min_true_score = min(true_scores) if true_scores else 0.0
    max_false_score = max(false_scores) if false_scores else 0.0
    score_separation = (
        min_true_score / max_false_score
        if max_false_score > 0.0 else math.inf)
    separation_passed = score_separation >= 3.0
    none_images = class_counts.get('NONE', 0)
    none_zero_candidates = all(
        not row['candidates'] for row in records
        if row['expected'] == 'NONE')
    none_validation = (
        'passed' if none_images > 0 and none_zero_candidates else
        'failed' if none_images > 0 else
        'not_accepted_no_negative_samples')

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = 0
    for record in records:
        expected = record['expected']
        candidates = record['candidates']
        winner = record['winner']
        actual = winner.color if winner is not None else 'NONE'
        wrong_accepted = [
            item for item in candidates
            if expected != 'NONE' and item.color != expected
        ]
        passed = (
            (expected == 'NONE' and not candidates) or
            (expected != 'NONE' and actual == expected and
             not wrong_accepted)
        )
        failures += int(not passed)
        target_dir = output / expected
        target_dir.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(record['path']), cv2.IMREAD_COLOR)
        rendered = annotate(image, record['proposals'], winner)
        cv2.imwrite(str(target_dir / record['path'].name), rendered)
        proposal_data = []
        for index, item in enumerate(record['proposals'], 1):
            metrics = {
                name: round(value, 6)
                for name, value in detection_metrics(item).items()
            }
            proposal_data.append({
                'rank': index,
                'accepted': item.score >= config.min_score,
                'color': item.color,
                'confidence': round(item.confidence, 6),
                'score': round(item.score, 6),
                'dots': item.dot_count,
                'length': round(item.length, 3),
                'residual': round(item.residual, 3),
                'spacing_cv': round(item.spacing_cv, 6),
                'metrics': metrics,
            })
        rows.append({
            'image': str(record['path'].relative_to(dataset)),
            'expected': expected,
            'actual': actual,
            'passed': passed,
            'candidate_count': len(candidates),
            'proposal_count': len(record['proposals']),
            'processing_ms': round(record['processing_ms'], 3),
            'winner_margin': record['margin'],
            'winner_score': None if winner is None else round(winner.score, 6),
            'candidate_scores': [
                {
                    'color': item.color,
                    'score': round(item.score, 6),
                }
                for item in candidates
            ],
            'proposals': proposal_data,
        })

    sorted_durations = sorted(durations)
    median_ms = (
        sorted_durations[len(sorted_durations) // 2] * 1000.0
        if sorted_durations else 0.0)
    p95_index = max(0, math.ceil(len(sorted_durations) * 0.95) - 1)
    p95_ms = (
        sorted_durations[p95_index] * 1000.0
        if sorted_durations else 0.0)
    summary = {
        'dataset': str(dataset),
        'config': str(config_path),
        'processing_scale': processing_scale,
        'images': len(rows),
        'class_counts': class_counts,
        'passed': len(rows) - failures,
        'failed': failures,
        'min_true_score': min_true_score,
        'max_false_score': max_false_score,
        'score_separation': score_separation,
        'required_score_separation': 3.0,
        'separation_passed': separation_passed,
        'none_validation': none_validation,
        'median_processing_ms': median_ms,
        'p95_processing_ms': p95_ms,
        'processing_target_p95_ms': 50.0,
        'processing_required_p95_ms': 60.0,
        'processing_required_passed': p95_ms < 60.0,
        'results': rows,
    }
    with (output / 'results.json').open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    with (output / 'results.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            'image', 'expected', 'actual', 'passed', 'candidate_count',
            'proposal_count', 'processing_ms', 'winner_margin',
            'winner_score', 'candidate_scores', 'proposals'))
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat['proposals'] = json.dumps(
                row['proposals'], ensure_ascii=False)
            flat['candidate_scores'] = json.dumps(
                row['candidate_scores'], ensure_ascii=False)
            writer.writerow(flat)
    print(
        f'evaluated={len(rows)} passed={len(rows)-failures} failed={failures} '
        f'scale={processing_scale:.2f} '
        f'min_true={min_true_score:.6f} max_false={max_false_score:.6f} '
        f'separation={score_separation:.3f} separation_ok={separation_passed} '
        f'none_validation={none_validation} '
        f'median_ms={median_ms:.1f} p95_ms={p95_ms:.1f} output={output}')
    return 0 if _evaluation_passed(
        failures, separation_passed, p95_ms, none_validation) else 1


def _scale_output(output: Path, scale: float, multiple: bool) -> Path:
    if not multiple:
        return output
    return output / f'scale_{scale:.2f}'.replace('.', '_')


def evaluate_scales(
        dataset: Path,
        output: Path,
        config_path: Path,
        processing_scales: Sequence[float]) -> int:
    scales = []
    for scale in processing_scales:
        normalized = max(0.1, min(float(scale), 1.0))
        if normalized not in scales:
            scales.append(normalized)
    multiple = len(scales) > 1
    results = [
        evaluate(dataset, _scale_output(output, scale, multiple), config_path, scale)
        for scale in scales
    ]
    return 0 if any(code == 0 for code in results) else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate and annotate the R2 LED dataset')
    parser.add_argument(
        '--camera-profile',
        choices=CAMERA_PROFILES,
        default=DEFAULT_CAMERA_PROFILE,
        help='选择独立的数据集和 detector 配置。')
    parser.add_argument(
        '--dataset', type=Path,
        help='覆盖默认数据集路径；默认 camera_data/<camera-profile>。')
    parser.add_argument(
        '--output', type=Path,
        help='覆盖默认输出路径；默认 camera_results/<camera-profile>。')
    parser.add_argument(
        '--config', type=Path,
        help='覆盖默认 detector；默认 config/cameras/<camera-profile>/detector.yaml。')
    parser.add_argument(
        '--processing-scale', type=float,
        help='离线评估缩放比例；默认使用 detector.yaml processing.scale 或 1.0。')
    parser.add_argument(
        '--scan-processing-scales', action='store_true',
        help='同时评估 1.0、0.75、0.67 三个缩放比例。')
    args = parser.parse_args()
    dataset = args.dataset or dataset_path(args.camera_profile)
    output = args.output or results_path(args.camera_profile)
    config = args.config or detector_config_path(args.camera_profile)
    if args.config is None:
        config = require_calibrated_detector(config, args.camera_profile)
    loaded = load_config(str(config))
    if args.scan_processing_scales:
        scales = (1.0, 0.75, 0.67)
    else:
        scales = (args.processing_scale or loaded.processing_scale,)
    raise SystemExit(evaluate_scales(
        dataset.resolve(),
        output.resolve(),
        config.resolve(),
        scales))


if __name__ == '__main__':
    main()
