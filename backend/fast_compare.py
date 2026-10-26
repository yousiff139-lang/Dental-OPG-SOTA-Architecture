import os
import sys
import json
import cv2
import numpy as np

backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)
from benchmark_suite import load_dataset_index, select_benchmark_cohorts, match_detections_to_ground_truth
from app.services.local_yolo_service import LocalYoloService

y8_path = os.path.join(backend_dir, "models", "weights", "sentoz_dental_opg.onnx")
y11_path = os.path.join(backend_dir, "models", "weights", "liodon_dental.onnx")

net_y8 = cv2.dnn.readNetFromONNX(y8_path)
net_y11 = cv2.dnn.readNetFromONNX(y11_path)

index = load_dataset_index()
cohorts = select_benchmark_cohorts(index, limit=3)
sids = []
for c in cohorts.values():
    sids.extend(c)

def run_onnx(net, img, class_map, conf_thresh=0.25):
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
    net.setInput(blob)
    preds = net.forward()[0].T
    boxes, confs, cids = [], [], []
    for row in preds:
        scores = row[4:]
        cid = int(np.argmax(scores))
        score = float(scores[cid])
        if score >= conf_thresh and cid in class_map:
            cx, cy, bw, bh = row[:4]
            orig_cx = (cx - left) / scale
            orig_cy = (cy - top) / scale
            orig_w = bw / scale
            orig_h = bh / scale
            x1 = max(0, min(orig_cx - orig_w / 2, w - 1))
            y1 = max(0, min(orig_cy - orig_h / 2, h - 1))
            boxes.append([int(x1), int(y1), int(round(orig_w)), int(round(orig_h))])
            confs.append(score)
            cids.append(cid)
    if not boxes:
        return []
    inds = cv2.dnn.NMSBoxes(boxes, confs, conf_thresh, 0.35)
    out = []
    if len(inds) > 0:
        for i in inds.flatten():
            bx, by, bw, bh = boxes[i]
            out.append({
                "class": class_map[cids[i]],
                "confidence": confs[i],
                "norm_box": (float(bx + bw / 2) / w, float(by + bh / 2) / h, float(bw) / w, float(bh) / h),
            })
    return out

map_y8 = {0: "Cavity", 1: "Cavity", 4: "Impacted Tooth", 5: "Periapical Lesion"}
map_y11 = {0: "Cavity", 1: "Periapical Lesion", 2: "Impacted Tooth"}

# 1. Standalone YOLOv8
# 2. Standalone YOLO11 (DENTEX)
# 3. Local YOLO Ensemble + Anatomical Structure Zoning Architecture (Stage 1+2 without cloud)
local_yolo = LocalYoloService()

res_y8 = {"tp_c": 0, "fp_c": 0, "tp_p": 0, "fp_p": 0, "clean_fps": 0, "clean_total": 0, "clean_correct": 0, "times": []}
res_y11 = {"tp_c": 0, "fp_c": 0, "tp_p": 0, "fp_p": 0, "clean_fps": 0, "clean_total": 0, "clean_correct": 0, "times": []}
res_arch = {"tp_c": 0, "fp_c": 0, "tp_p": 0, "fp_p": 0, "clean_fps": 0, "clean_total": 0, "clean_correct": 0, "times": []}

total_gt_c = 0
total_gt_p = 0

for sid in sids:
    meta = index[sid]
    img = cv2.imread(meta["image_path"])
    h, w = img.shape[:2]
    gt = meta["boxes"]
    is_clean = meta["classes"].issubset({5, 2})
    gt_c = len([b for b in gt if b["class_id"] == 0])
    gt_p = len([b for b in gt if b["class_id"] == 1])
    total_gt_c += gt_c
    total_gt_p += gt_p

    # y8 standalone
    t0 = cv2.getTickCount()
    p8 = run_onnx(net_y8, img, map_y8, conf_thresh=0.25)
    t_y8 = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
    res_y8["times"].append(t_y8)
    tp_c, fp_c, _ = match_detections_to_ground_truth(p8, gt, 0)
    tp_p, fp_p, _ = match_detections_to_ground_truth(p8, gt, 1)
    res_y8["tp_c"] += tp_c
    res_y8["fp_c"] += fp_c
    res_y8["tp_p"] += tp_p
    res_y8["fp_p"] += fp_p
    if is_clean:
        fps = len([p for p in p8 if p["class"] in ("Cavity", "Periapical Lesion")])
        res_y8["clean_total"] += 1
        res_y8["clean_fps"] += fps
        if fps == 0:
            res_y8["clean_correct"] += 1

    # y11 standalone
    t0 = cv2.getTickCount()
    p11 = run_onnx(net_y11, img, map_y11, conf_thresh=0.25)
    t_y11 = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
    res_y11["times"].append(t_y11)
    tp_c, fp_c, _ = match_detections_to_ground_truth(p11, gt, 0)
    tp_p, fp_p, _ = match_detections_to_ground_truth(p11, gt, 1)
    res_y11["tp_c"] += tp_c
    res_y11["fp_c"] += fp_c
    res_y11["tp_p"] += tp_p
    res_y11["fp_p"] += fp_p
    if is_clean:
        fps = len([p for p in p11 if p["class"] in ("Cavity", "Periapical Lesion")])
        res_y11["clean_total"] += 1
        res_y11["clean_fps"] += fps
        if fps == 0:
            res_y11["clean_correct"] += 1

    # with architecture (anatomical zoning & consensus)
    t0 = cv2.getTickCount()
    parch_raw = local_yolo.detect(img, conf_threshold=0.25, is_panoramic=True)
    t_arch = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
    res_arch["times"].append(t_arch)
    parch = []
    for p in parch_raw:
        parch.append({
            "class": p["class"],
            "confidence": p["confidence"],
            "norm_box": (p["x"] / w, p["y"] / h, p["width"] / w, p["height"] / h),
        })
    tp_c, fp_c, _ = match_detections_to_ground_truth(parch, gt, 0)
    tp_p, fp_p, _ = match_detections_to_ground_truth(parch, gt, 1)
    res_arch["tp_c"] += tp_c
    res_arch["fp_c"] += fp_c
    res_arch["tp_p"] += tp_p
    res_arch["fp_p"] += fp_p
    if is_clean:
        fps = len([p for p in parch if p["class"] in ("Cavity", "Periapical Lesion")])
        res_arch["clean_total"] += 1
        res_arch["clean_fps"] += fps
        if fps == 0:
            res_arch["clean_correct"] += 1

print("\n" + "="*80)
print("BENCHMARK MATRIX: STANDALONE YOLOv8 vs STANDALONE YOLO11 vs WITH ARCHITECTURE")
print("="*80)
print(f"Ground Truth Totals -> Caries: {total_gt_c}, Periapical Lesions: {total_gt_p}, Clean Images: {res_y8['clean_total']}\n")

models = [
    ("1. Standalone YOLOv8 (Singly without architecture)", res_y8),
    ("2. Standalone YOLO11 (Singly without architecture)", res_y11),
    ("3. Multi-Scanner Architectural Pipeline (Anatomy Zoning + Consensus)", res_arch),
]

for name, r in models:
    prec_c = (r["tp_c"] / (r["tp_c"] + r["fp_c"]) * 100) if (r["tp_c"] + r["fp_c"]) > 0 else 0
    rec_c = (r["tp_c"] / total_gt_c * 100) if total_gt_c > 0 else 0
    prec_p = (r["tp_p"] / (r["tp_p"] + r["fp_p"]) * 100) if (r["tp_p"] + r["fp_p"]) > 0 else 0
    rec_p = (r["tp_p"] / total_gt_p * 100) if total_gt_p > 0 else 0
    spec = (r["clean_correct"] / r["clean_total"] * 100) if r["clean_total"] > 0 else 0
    avg_t = sum(r["times"]) / len(r["times"]) if r["times"] else 0

    print(f"--- {name} ---")
    print(f"  Clean Film Specificity:        {spec:.1f}% ({r['clean_correct']}/{r['clean_total']} clean films with 0 false alarms)")
    print(f"  False Positives on Clean Film: {r['clean_fps']} total ({r['clean_fps']/r['clean_total']:.2f} per clean film)")
    print(f"  Caries Precision:              {prec_c:.1f}% (TP={r['tp_c']}, FP={r['fp_c']})")
    print(f"  Caries Recall (Sensitivity):   {rec_c:.1f}%")
    print(f"  Periapical Lesion Precision:   {prec_p:.1f}% (TP={r['tp_p']}, FP={r['fp_p']})")
    print(f"  Periapical Lesion Recall:      {rec_p:.1f}%")
    print(f"  Mean Inference Latency:        {avg_t:.2f} ms")
    print()
