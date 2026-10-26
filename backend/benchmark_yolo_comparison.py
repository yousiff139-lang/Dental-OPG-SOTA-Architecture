"""
Comprehensive Benchmark: YOLOv8 vs YOLO11 (DENTEX)
Evaluating:
1. Standalone Raw YOLOv8 (Singly without architecture)
2. Standalone Raw YOLO11 (Singly without architecture)
3. YOLOv8 + Multi-Scanning Architecture Pipeline
4. YOLO11 + Multi-Scanning Architecture Pipeline
"""

import os
import sys
import time
import json
import asyncio
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.multi_scanner_service import MultiScanningService
from app.services.local_yolo_service import LocalYoloService
from benchmark_suite import (
    load_dataset_index,
    select_benchmark_cohorts,
    compute_box_iou,
    match_detections_to_ground_truth,
    CLASS_MAPPING,
)

WEIGHTS_DIR = os.path.join(backend_dir, "models", "weights")
YOLO8_OPG_PATH = os.path.join(WEIGHTS_DIR, "sentoz_dental_opg.onnx")
YOLO8_CARIES_PATH = os.path.join(WEIGHTS_DIR, "gegesay_caries_yolov8s.onnx")
YOLO11_DENTEX_PATH = os.path.join(WEIGHTS_DIR, "liodon_dental.onnx")


class StandaloneYoloDetector:
    """Runs a single YOLO model without any pipeline or anatomical filters."""
    def __init__(self, model_type: str):
        self.model_type = model_type
        if model_type == "yolo8":
            self.net = cv2.dnn.readNetFromONNX(YOLO8_OPG_PATH)
            # sentoz classes: 0: BDC-BDR, 1: Caries, 2: Fractured, 3: Healthy, 4: Impacted, 5: Infection
            self.class_map = {
                0: "Cavity",
                1: "Cavity",
                4: "Impacted Tooth",
                5: "Periapical Lesion",
            }
        elif model_type == "yolo11":
            self.net = cv2.dnn.readNetFromONNX(YOLO11_DENTEX_PATH)
            # liodon classes: 0: caries, 1: periapical_lesion, 2: impacted_tooth
            self.class_map = {
                0: "Cavity",
                1: "Periapical Lesion",
                2: "Impacted Tooth",
            }
        else:
            raise ValueError(f"Unknown model_type {model_type}")

    def detect(self, img: np.ndarray, conf_thresh: float = 0.25, iou_thresh: float = 0.35) -> List[Dict[str, Any]]:
        h, w = img.shape[:2]
        input_size = 640
        scale = min(input_size / h, input_size / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        top = (input_size - nh) // 2
        left = (input_size - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (input_size, input_size), swapRB=True)
        self.net.setInput(blob)
        preds = self.net.forward()[0].T

        boxes = []
        confidences = []
        class_ids = []

        for row in preds:
            scores = row[4:]
            cid = int(np.argmax(scores))
            score = float(scores[cid])
            if score >= conf_thresh and cid in self.class_map:
                cx, cy, bw, bh = row[:4]
                orig_cx = (cx - left) / scale
                orig_cy = (cy - top) / scale
                orig_w = bw / scale
                orig_h = bh / scale
                x1 = max(0, min(orig_cx - orig_w / 2, w - 1))
                y1 = max(0, min(orig_cy - orig_h / 2, h - 1))
                boxes.append([int(x1), int(y1), int(round(orig_w)), int(round(orig_h))])
                confidences.append(score)
                class_ids.append(cid)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, iou_thresh)
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = boxes[idx]
                cid = class_ids[idx]
                results.append({
                    "class": self.class_map[cid],
                    "confidence": round(float(confidences[idx]), 4),
                    "x": float(bx + bw / 2),
                    "y": float(by + bh / 2),
                    "width": float(bw),
                    "height": float(bh),
                    "norm_box": (
                        float(bx + bw / 2) / w,
                        float(by + bh / 2) / h,
                        float(bw) / w,
                        float(bh) / h,
                    ),
                })
        return results


async def run_comparison_benchmark():
    index = load_dataset_index()
    cohorts = select_benchmark_cohorts(index, limit=3)
    
    # 12 benchmark samples
    all_sample_ids = []
    for c_name, sids in cohorts.items():
        all_sample_ids.extend(sids)

    print(f"Total benchmark samples: {len(all_sample_ids)} across 4 cohorts:")
    for c_name, sids in cohorts.items():
        print(f"  {c_name}: {sids}")

    # Initialize detectors
    yolo8_standalone = StandaloneYoloDetector("yolo8")
    yolo11_standalone = StandaloneYoloDetector("yolo11")

    # Multi-scanner pipelines:
    # We test pipeline with YOLO8 vs pipeline with YOLO11
    scanner = MultiScanningService()

    eval_data = {
        "yolo8_standalone": {"tp_caries": 0, "fp_caries": 0, "fn_caries": 0, "tp_pa": 0, "fp_pa": 0, "fn_pa": 0, "clean_fp": 0, "clean_total": 0, "clean_correct": 0, "times": []},
        "yolo11_standalone": {"tp_caries": 0, "fp_caries": 0, "fn_caries": 0, "tp_pa": 0, "fp_pa": 0, "fn_pa": 0, "clean_fp": 0, "clean_total": 0, "clean_correct": 0, "times": []},
        "yolo11_pipeline": {"tp_caries": 0, "fp_caries": 0, "fn_caries": 0, "tp_pa": 0, "fp_pa": 0, "fn_pa": 0, "clean_fp": 0, "clean_total": 0, "clean_correct": 0, "times": []},
    }

    gt_caries_total = 0
    gt_pa_total = 0

    print("\nStarting Benchmark Runs on 12 Datasets...\n")

    for sid in all_sample_ids:
        meta = index[sid]
        img_path = meta["image_path"]
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        gt_boxes = meta["boxes"]
        is_clean = meta["classes"].issubset({5, 2})

        # Count ground truth
        gt_caries = len([b for b in gt_boxes if b["class_id"] == 0])
        gt_pa = len([b for b in gt_boxes if b["class_id"] == 1])
        gt_caries_total += gt_caries
        gt_pa_total += gt_pa

        print(f"\n---> Evaluating Sample #{sid} (Clean={is_clean}, Caries={gt_caries}, PA={gt_pa}):")

        # 1. Standalone YOLOv8
        t0 = time.time()
        preds_y8 = yolo8_standalone.detect(img, conf_thresh=0.25)
        dt_y8 = (time.time() - t0) * 1000
        eval_data["yolo8_standalone"]["times"].append(dt_y8)

        tp_c8, fp_c8, fn_c8 = match_detections_to_ground_truth(preds_y8, gt_boxes, 0)
        tp_p8, fp_p8, fn_p8 = match_detections_to_ground_truth(preds_y8, gt_boxes, 1)
        eval_data["yolo8_standalone"]["tp_caries"] += tp_c8
        eval_data["yolo8_standalone"]["fp_caries"] += fp_c8
        eval_data["yolo8_standalone"]["fn_caries"] += fn_c8
        eval_data["yolo8_standalone"]["tp_pa"] += tp_p8
        eval_data["yolo8_standalone"]["fp_pa"] += fp_p8
        eval_data["yolo8_standalone"]["fn_pa"] += fn_p8
        if is_clean:
            fps = len([p for p in preds_y8 if p["class"] in ("Cavity", "Periapical Lesion")])
            eval_data["yolo8_standalone"]["clean_total"] += 1
            eval_data["yolo8_standalone"]["clean_fp"] += fps
            if fps == 0:
                eval_data["yolo8_standalone"]["clean_correct"] += 1

        print(f"  [YOLOv8 Standalone] {len(preds_y8)} preds in {dt_y8:.1f}ms (TP_Caries={tp_c8}, FP_Caries={fp_c8}, TP_PA={tp_p8}, FP_PA={fp_p8})")

        # 2. Standalone YOLO11 (DENTEX)
        t0 = time.time()
        preds_y11 = yolo11_standalone.detect(img, conf_thresh=0.25)
        dt_y11 = (time.time() - t0) * 1000
        eval_data["yolo11_standalone"]["times"].append(dt_y11)

        tp_c11, fp_c11, fn_c11 = match_detections_to_ground_truth(preds_y11, gt_boxes, 0)
        tp_p11, fp_p11, fn_p11 = match_detections_to_ground_truth(preds_y11, gt_boxes, 1)
        eval_data["yolo11_standalone"]["tp_caries"] += tp_c11
        eval_data["yolo11_standalone"]["fp_caries"] += fp_c11
        eval_data["yolo11_standalone"]["fn_caries"] += fn_c11
        eval_data["yolo11_standalone"]["tp_pa"] += tp_p11
        eval_data["yolo11_standalone"]["fp_pa"] += fp_p11
        eval_data["yolo11_standalone"]["fn_pa"] += fn_p11
        if is_clean:
            fps = len([p for p in preds_y11 if p["class"] in ("Cavity", "Periapical Lesion")])
            eval_data["yolo11_standalone"]["clean_total"] += 1
            eval_data["yolo11_standalone"]["clean_fp"] += fps
            if fps == 0:
                eval_data["yolo11_standalone"]["clean_correct"] += 1

        print(f"  [YOLO11 Standalone] {len(preds_y11)} preds in {dt_y11:.1f}ms (TP_Caries={tp_c11}, FP_Caries={fp_c11}, TP_PA={tp_p11}, FP_PA={fp_p11})")

        # 3. Pipeline Run
        t0 = time.time()
        res_pipe = await scanner.scan_dental_image(img_path)
        dt_pipe = (time.time() - t0) * 1000
        eval_data["yolo11_pipeline"]["times"].append(dt_pipe)

        pipe_preds = []
        for p in res_pipe.get("predictions", []):
            pipe_preds.append({
                "class": p.get("class"),
                "confidence": p.get("confidence", 0.0),
                "norm_box": (
                    p.get("x", 0.0) / w,
                    p.get("y", 0.0) / h,
                    p.get("width", 0.0) / w,
                    p.get("height", 0.0) / h,
                ),
            })

        tp_cp, fp_cp, fn_cp = match_detections_to_ground_truth(pipe_preds, gt_boxes, 0)
        tp_pp, fp_pp, fn_pp = match_detections_to_ground_truth(pipe_preds, gt_boxes, 1)
        eval_data["yolo11_pipeline"]["tp_caries"] += tp_cp
        eval_data["yolo11_pipeline"]["fp_caries"] += fp_cp
        eval_data["yolo11_pipeline"]["fn_caries"] += fn_cp
        eval_data["yolo11_pipeline"]["tp_pa"] += tp_pp
        eval_data["yolo11_pipeline"]["fp_pa"] += fp_pp
        eval_data["yolo11_pipeline"]["fn_pa"] += fn_pp
        if is_clean:
            fps = len([p for p in pipe_preds if p["class"] in ("Cavity", "Periapical Lesion")])
            eval_data["yolo11_pipeline"]["clean_total"] += 1
            eval_data["yolo11_pipeline"]["clean_fp"] += fps
            if fps == 0 or res_pipe.get("is_healthy", False):
                eval_data["yolo11_pipeline"]["clean_correct"] += 1

        print(f"  [Multi-Scanner Pipeline] {len(pipe_preds)} preds in {dt_pipe:.1f}ms (TP_Caries={tp_cp}, FP_Caries={fp_cp}, TP_PA={tp_pp}, FP_PA={fp_pp}, is_healthy={res_pipe.get('is_healthy')})")

    # Final summary calculations
    print("\n" + "=" * 80)
    print("                     FINAL BENCHMARK COMPARISON MATRIX")
    print("=" * 80)

    summary = {}
    for model_key in ["yolo8_standalone", "yolo11_standalone", "yolo11_pipeline"]:
        d = eval_data[model_key]
        tp_c = d["tp_caries"]
        fp_c = d["fp_caries"]
        prec_c = (tp_c / (tp_c + fp_c) * 100) if (tp_c + fp_c) > 0 else 0.0
        rec_c = (tp_c / gt_caries_total * 100) if gt_caries_total > 0 else 0.0

        tp_p = d["tp_pa"]
        fp_p = d["fp_pa"]
        prec_p = (tp_p / (tp_p + fp_p) * 100) if (tp_p + fp_p) > 0 else 0.0
        rec_p = (tp_p / gt_pa_total * 100) if gt_pa_total > 0 else 0.0

        spec = (d["clean_correct"] / d["clean_total"] * 100) if d["clean_total"] > 0 else 0.0
        avg_time = sum(d["times"]) / len(d["times"]) if d["times"] else 0.0

        summary[model_key] = {
            "clean_specificity_pct": round(spec, 2),
            "clean_fps_total": d["clean_fp"],
            "caries_precision_pct": round(prec_c, 2),
            "caries_recall_pct": round(rec_c, 2),
            "pa_precision_pct": round(prec_p, 2),
            "pa_recall_pct": round(rec_p, 2),
            "mean_latency_ms": round(avg_time, 2),
        }

    print(json.dumps(summary, indent=2))
    with open(os.path.join(backend_dir, "yolo_comparison_results.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

if __name__ == "__main__":
    asyncio.run(run_comparison_benchmark())
