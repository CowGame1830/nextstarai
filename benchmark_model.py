import argparse
import json
import statistics
import os
import random
import time
from collections import defaultdict
from datetime import datetime

import cv2
from ultralytics import YOLO


def get_class_name(names_map, class_id):
    if isinstance(names_map, dict):
        return str(names_map.get(class_id, class_id))
    if isinstance(names_map, list) and 0 <= class_id < len(names_map):
        return str(names_map[class_id])
    return str(class_id)


def _accumulate_result_stats(result, names_map, class_counts, class_conf_sum, class_frame_hits, all_conf_scores):
    seen_classes = set()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return

    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()

    for class_id, score in zip(cls_ids, confs):
        class_name = get_class_name(names_map, int(class_id))
        class_counts[class_name] += 1
        class_conf_sum[class_name] += float(score)
        all_conf_scores.append(float(score))
        seen_classes.add(class_name)

    for class_name in seen_classes:
        class_frame_hits[class_name] += 1


def summarize_video_performance(model, video_path, conf=0.1, imgsz=1280, batch_size=8, max_frames=None, lazy_load=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    names_map = model.names if hasattr(model, "names") else {}

    total_frames_processed = 0
    inference_seconds = 0.0

    class_counts = defaultdict(int)
    class_conf_sum = defaultdict(float)
    class_frame_hits = defaultdict(int)
    all_conf_scores = []

    frame_buffer = []

    def flush_batch():
        nonlocal inference_seconds
        if not frame_buffer:
            return

        start = time.perf_counter()
        results = model.predict(frame_buffer, conf=conf, imgsz=imgsz, verbose=False)
        inference_seconds += time.perf_counter() - start

        for result in results:
            _accumulate_result_stats(
                result,
                names_map,
                class_counts,
                class_conf_sum,
                class_frame_hits,
                all_conf_scores,
            )

        frame_buffer.clear()

    if lazy_load:
        stream_start = time.perf_counter()
        for result in model.predict(source=video_path, conf=conf, imgsz=imgsz, verbose=False, stream=True):
            total_frames_processed += 1

            _accumulate_result_stats(
                result,
                names_map,
                class_counts,
                class_conf_sum,
                class_frame_hits,
                all_conf_scores,
            )

            speed = getattr(result, "speed", None)
            if isinstance(speed, dict) and "inference" in speed:
                inference_seconds += float(speed["inference"]) / 1000.0

            if max_frames and total_frames_processed >= max_frames:
                break

        if inference_seconds <= 0:
            inference_seconds = time.perf_counter() - stream_start
    else:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            total_frames_processed += 1
            frame_buffer.append(frame)

            if len(frame_buffer) >= batch_size:
                flush_batch()

            if max_frames and total_frames_processed >= max_frames:
                break

        flush_batch()
    cap.release()

    if total_frames_processed == 0:
        raise RuntimeError("No frames were processed from the video.")

    total_detections = int(sum(class_counts.values()))

    per_class = {}
    for class_name in sorted(class_counts.keys()):
        detections = class_counts[class_name]
        avg_conf = class_conf_sum[class_name] / detections if detections else 0.0
        frame_presence_ratio = class_frame_hits[class_name] / total_frames_processed

        per_class[class_name] = {
            "detections": int(detections),
            "avg_detections_per_frame": round(detections / total_frames_processed, 4),
            "avg_confidence": round(avg_conf, 4),
            "avg_confidence_percent": round(avg_conf * 100.0, 2),
            "frame_presence_ratio": round(frame_presence_ratio, 4),
        }

    confidence_summary = {
        "mean": 0.0,
        "mean_percent": 0.0,
        "median": 0.0,
        "median_percent": 0.0,
    }
    if all_conf_scores:
        mean_conf = statistics.fmean(all_conf_scores)
        median_conf = statistics.median(all_conf_scores)
        confidence_summary = {
            "mean": round(mean_conf, 4),
            "mean_percent": round(mean_conf * 100.0, 2),
            "median": round(median_conf, 4),
            "median_percent": round(median_conf * 100.0, 2),
        }

    fps_inference_only = (total_frames_processed / inference_seconds) if inference_seconds > 0 else 0.0

    return {
        "video_path": video_path,
        "video_fps": round(video_fps, 4),
        "video_total_frames": total_frames_in_video,
        "frames_processed": total_frames_processed,
        "batch_size": batch_size,
        "lazy_load": lazy_load,
        "max_frames": max_frames,
        "inference_seconds": round(inference_seconds, 4),
        "fps_inference_only": round(fps_inference_only, 4),
        "ms_per_frame_inference": round((inference_seconds / total_frames_processed) * 1000.0, 4),
        "total_detections": total_detections,
        "avg_detections_per_frame": round(total_detections / total_frames_processed, 4),
        "confidence_summary": confidence_summary,
        "per_class": per_class,
    }


def summarize_validation(model, data_yaml, imgsz=1280, conf=0.1):
    start = time.perf_counter()
    metrics = model.val(data=data_yaml, imgsz=imgsz, conf=conf, verbose=False)
    elapsed = time.perf_counter() - start

    summary = {
        "data_yaml": data_yaml,
        "validation_seconds": round(elapsed, 4),
    }

    if hasattr(metrics, "box"):
        box_metrics = metrics.box
        precision = float(getattr(box_metrics, "mp", 0.0))
        recall = float(getattr(box_metrics, "mr", 0.0))
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        map50 = float(getattr(box_metrics, "map50", 0.0))
        map50_95 = float(getattr(box_metrics, "map", 0.0))

        summary["precision"] = round(precision, 4)
        summary["precision_percent"] = round(precision * 100.0, 2)
        summary["recall"] = round(recall, 4)
        summary["recall_percent"] = round(recall * 100.0, 2)
        summary["f1_score"] = round(f1, 4)
        summary["mAP50_95"] = round(map50_95, 4)
        summary["mAP50"] = round(map50, 4)
        summary["mAP@50"] = round(map50, 4)
        summary["mAP75"] = round(float(getattr(box_metrics, "map75", 0.0)), 4)
        summary["metrics"] = {
            "precision": summary["precision"],
            "recall": summary["recall"],
            "mAP50": summary["mAP50"],
            "mAP@50": summary["mAP@50"],
            "mAP50_95": summary["mAP50_95"],
            "f1_score": summary["f1_score"],
        }

        if hasattr(box_metrics, "maps"):
            maps = box_metrics.maps
            if hasattr(maps, "tolist"):
                summary["per_class_mAP50_95"] = [round(float(v), 4) for v in maps.tolist()]

    return summary


def _list_video_files(input_dir):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input videos directory not found: {input_dir}")

    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    video_paths = []
    for name in os.listdir(input_dir):
        full_path = os.path.join(input_dir, name)
        if os.path.isfile(full_path) and os.path.splitext(name)[1].lower() in exts:
            video_paths.append(full_path)

    if not video_paths:
        raise RuntimeError(f"No video files found in: {input_dir}")

    return sorted(video_paths)


def _pick_random_videos(video_paths, random_count, random_seed=None):
    if random_count <= 0:
        raise ValueError("random_count must be greater than zero")

    count = min(random_count, len(video_paths))
    rng = random.Random(random_seed)
    return rng.sample(video_paths, count)


def _aggregate_video_reports(video_reports):
    if not video_reports:
        return {}

    frame_total = sum(v["frames_processed"] for v in video_reports)
    det_total = sum(v["total_detections"] for v in video_reports)
    sec_total = sum(v["inference_seconds"] for v in video_reports)
    fps_values = [v["fps_inference_only"] for v in video_reports if v["fps_inference_only"] > 0]

    return {
        "videos_processed": len(video_reports),
        "frames_processed_total": frame_total,
        "inference_seconds_total": round(sec_total, 4),
        "fps_inference_overall": round((frame_total / sec_total), 4) if sec_total > 0 else 0.0,
        "fps_inference_mean_of_videos": round(statistics.fmean(fps_values), 4) if fps_values else 0.0,
        "total_detections": int(det_total),
        "avg_detections_per_frame": round((det_total / frame_total), 4) if frame_total > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO model on football video and optional validation set")
    parser.add_argument("--model", default="yolo11n.pt", help="Path to model weights")
    parser.add_argument("--video", default=None, help="Path to a single test video (overrides random video selection)")
    parser.add_argument("--input-dir", default="input_videos", help="Directory containing candidate videos")
    parser.add_argument("--random-count", type=int, default=10, help="Number of random videos to benchmark from input-dir")
    parser.add_argument("--random-seed", type=int, default=None, help="Optional seed for reproducible random video selection")
    parser.add_argument("--data", default=None, help="Optional data.yaml for labeled validation")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for video inference")
    parser.add_argument("--lazy-load", action="store_true", help="Use Ultralytics stream mode for lazy frame loading")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for quick tests")
    parser.add_argument("--output-dir", default="output_data", help="Directory for benchmark JSON")

    args = parser.parse_args()

    model = YOLO(args.model)

    if args.video:
        selected_videos = [args.video]
    else:
        video_pool = _list_video_files(args.input_dir)
        selected_videos = _pick_random_videos(video_pool, args.random_count, args.random_seed)

    video_reports = []
    for video_path in selected_videos:
        video_reports.append(
            summarize_video_performance(
                model=model,
                video_path=video_path,
                conf=args.conf,
                imgsz=args.imgsz,
                batch_size=args.batch_size,
                max_frames=args.max_frames,
                lazy_load=args.lazy_load,
            )
        )

    report = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_path": args.model,
        "video_selection": {
            "video_count": len(selected_videos),
            "input_dir": args.input_dir,
            "random_count_requested": args.random_count,
            "random_seed": args.random_seed,
            "used_single_video_arg": bool(args.video),
        },
        "video_benchmarks": video_reports,
        "video_benchmark_summary": _aggregate_video_reports(video_reports),
    }

    if args.data:
        try:
            report["validation_benchmark"] = summarize_validation(
                model=model,
                data_yaml=args.data,
                imgsz=args.imgsz,
                conf=args.conf,
            )
        except FileNotFoundError as exc:
            report["validation_benchmark_error"] = str(exc)

    os.makedirs(args.output_dir, exist_ok=True)
    model_stem = os.path.splitext(os.path.basename(args.model))[0]
    out_path = os.path.join(args.output_dir, f"benchmark_{model_stem}_{report['timestamp']}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nBenchmark saved to: {out_path}")


if __name__ == "__main__":
    main()
