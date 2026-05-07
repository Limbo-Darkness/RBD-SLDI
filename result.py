import pre_processors
import controlled_degradations
import edge_detectors
import cv2
import numpy as np
from pathlib import Path
import csv
import io
 
# purpose of project is to evaluate entire dataset
 
 
def _mask_to_contour_edge(anno_gray):
    """Convert a filled segmentation mask to a binary edge map using contours.
 
    The annotated masks are filled regions (white interior, black background).
    Edge detectors produce boundary pixels, not filled regions, so a direct
    pixel comparison would be meaningless.  This function extracts the contour
    of the mask — i.e. just the boundary — so both sides of the comparison are
    in the same representation: edge pixels vs edge pixels.
 
    Returns a binary uint8 array (0 or 1) the same size as anno_gray.
    """
    # Binarise the mask first
    _, binary = cv2.threshold(anno_gray, 127, 255, cv2.THRESH_BINARY)
    # Find all external contours of the filled region
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Draw contours onto a blank canvas with thickness=1 for a clean 1-pixel boundary
    edge_map = np.zeros_like(anno_gray, dtype=np.uint8)
    cv2.drawContours(edge_map, contours, -1, color=1, thickness=1)
    return edge_map
 
 
# Two radii govern the evaluation:
#
# TOLERANCE_PX — matching radius.  A predicted edge within this many pixels
#   of the GT contour counts as a true positive (and vice-versa).  2px is
#   consistent with standard edge benchmarks (BSDS500).
#
# EVAL_BAND_PX — evaluation band half-width.  Only predicted edge pixels
#   within this distance of the GT contour are counted at all; edges deep
#   inside the lesion or far outside it are irrelevant to boundary detection
#   quality and would otherwise inflate the false-positive count enormously.
#   This is standard practice for boundary-specific evaluation in medical imaging.
#   Set to 10px (~1-2% of a typical ISIC image dimension).
TOLERANCE_PX = 2
EVAL_BAND_PX = 10
 
 
def _dilate(binary, radius):
    """Dilate a 0/1 or 0/255 binary array by *radius* pixels using an elliptical kernel."""
    k = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(binary.astype(np.uint8), kernel)
 
 
def _confusion(pred_bin, gt_bin, tolerance=TOLERANCE_PX, eval_band=EVAL_BAND_PX):
    """Return (TP, FP, FN, TN) using tolerance matching within an evaluation band.
 
    Predicted edge pixels are first masked to an evaluation band around the GT
    contour (radius = eval_band).  This discards edges that are irrelevant to the
    boundary detection task (internal texture, distant background edges) without
    penalising detectors for doing their job correctly inside the lesion.
 
    Within that band, a predicted pixel is a TP if it falls within *tolerance*
    pixels of the GT contour, and an FP otherwise.  GT pixels not covered by any
    predicted edge within *tolerance* are FNs.
    """
    # Build the evaluation band around the GT contour
    gt_band = _dilate(gt_bin, eval_band)        # 0/1 mask of the evaluation region
 
    # Restrict predicted edges to the evaluation band
    pred_in_band = np.logical_and(pred_bin == 1, gt_band == 1).astype(np.uint8)
 
    # Dilated zones for tolerance matching
    gt_dilated   = _dilate(gt_bin,       tolerance)  # zone around GT contour
    pred_dilated = _dilate(pred_in_band, tolerance)  # zone around band-masked predictions
 
    # TP: in-band predicted pixel within tolerance of GT contour
    tp = int(np.logical_and(pred_in_band == 1, gt_dilated == 1).sum())
    # FP: in-band predicted pixel outside GT tolerance zone
    fp = int(np.logical_and(pred_in_band == 1, gt_dilated == 0).sum())
    # FN: GT contour pixel not covered by any in-band prediction within tolerance
    fn = int(np.logical_and(gt_bin == 1, pred_dilated == 0).sum())
    # TN: non-edge pixel outside GT tolerance zone
    tn = int(np.logical_and(pred_bin == 0, gt_dilated == 0).sum())
    return tp, fp, fn, tn
 
 
def _metrics(tp, fp, fn, tn):
    """Derive IoU, precision, recall, and F1 from confusion counts."""
    union        = tp + fp + fn
    iou_score    = tp / union        if union > 0  else 1.0
    precision    = tp / (tp + fp)    if (tp + fp) > 0 else 0.0
    recall       = tp / (tp + fn)    if (tp + fn) > 0 else 0.0
    denom        = precision + recall
    f1           = 2 * precision * recall / denom if denom > 0 else 0.0
    return iou_score, precision, recall, f1
 
 
def _band_otsu(mag_f32, gt_bin):
    """Binarise a gradient magnitude image using an Otsu threshold computed only
    within the evaluation band around the GT contour.
 
    Parameters
    ----------
    mag_f32  : float32 array, values in [0, 255]
    gt_bin   : 0/1 uint8 array of the GT contour
 
    Returns
    -------
    0/1 uint8 binary edge map
 
    Rationale
    ---------
    On dermoscopic images, internal lesion texture and hair artefacts produce
    strong gradients that dominate a global histogram — Otsu then picks a low
    threshold and keeps thousands of irrelevant pixels.  Restricting the Otsu
    computation to the ±EVAL_BAND_PX zone around the expected boundary focuses
    the threshold on the gradient distribution where the lesion edge should be,
    reliably separating true boundary responses from background texture.
 
    If the image is already binary (Canny output: only values 0 and 255),
    the histogram is trivially bimodal and Otsu returns 127, preserving the map.
    """
    mag_u8 = np.clip(mag_f32, 0, 255).astype(np.uint8)
 
    gt_band = _dilate(gt_bin, EVAL_BAND_PX)
    band_pixels = mag_u8[gt_band == 1]
 
    if len(band_pixels) == 0 or band_pixels.max() == band_pixels.min():
        # Degenerate case: no band pixels or all same value — fall back to global Otsu
        thresh, _ = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Compute Otsu threshold from band histogram only
        hist, _ = np.histogram(band_pixels, bins=256, range=(0, 256))
        total = len(band_pixels)
        best_thresh, best_var = 0, -1
        w0, sum0 = 0, 0
        total_sum = np.dot(np.arange(256), hist)
        sum1 = total_sum
        for t in range(256):
            w0 += hist[t]
            w1 = total - w0
            if w0 == 0 or w1 == 0:
                continue
            sum0 += t * hist[t]
            sum1 -= t * hist[t]
            mu0 = sum0 / w0
            mu1 = sum1 / w1
            var = w0 * w1 * (mu0 - mu1) ** 2
            if var > best_var:
                best_var, best_thresh = var, t
        thresh = best_thresh
 
    return (mag_u8 >= thresh).astype(np.uint8)
 
 
def iou(edgeimg, annotatedimg, type):
    """Compare an edge-detected image against a contour-extracted annotation mask.
 
    edgeimg       -- path (str) or grayscale numpy array of the edge-detected image
    annotatedimg  -- path (str) or grayscale numpy array of the filled segmentation mask
 
    The annotation mask is converted to a 1-pixel contour boundary via
    cv2.findContours before comparison, so both sides are in the same
    representation (predicted edges vs ground-truth boundary).
 
    When type == "disp", a colour overlay is written to iou.png:
      - True  positives (TP) -> green   (edge correctly found)
      - False positives (FP) -> red     (edge predicted, not in GT boundary)
      - False negatives (FN) -> blue    (GT boundary pixel missed)
      - True  negatives (TN) -> black
 
    Returns a dict with keys: iou, precision, recall, f1
    """
    # --- load edge prediction (path or float32/uint8 array) ---
    if isinstance(edgeimg, str):
        edge = cv2.imread(edgeimg, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    else:
        edge = edgeimg.astype(np.float32)
 
    # --- load annotation and extract contour boundary ---
    anno_raw = cv2.imread(annotatedimg, cv2.IMREAD_GRAYSCALE) if isinstance(annotatedimg, str) else annotatedimg
    if anno_raw.shape != edge.shape[:2]:
        anno_raw = cv2.resize(anno_raw, (edge.shape[1], edge.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    gt = _mask_to_contour_edge(anno_raw)   # 0/1 boundary map
 
    # --- binarise edge map with band-restricted Otsu ---
    # Global thresholding (fixed or Otsu on the full image) fails on dermoscopic
    # images because internal texture produces strong gradients that dominate the
    # histogram. By computing Otsu only on the pixels within the evaluation band
    # around the GT contour, the threshold adapts to the gradient distribution
    # specifically near the lesion boundary — where it matters for scoring.
    # Canny already returns a binary 0/255 map, so band-Otsu is a no-op for it.
    pred = _band_otsu(edge, gt)
 
    # --- confusion matrix & metrics ---
    tp, fp, fn, tn = _confusion(pred, gt)
    iou_score, precision, recall, f1 = _metrics(tp, fp, fn, tn)
 
    # --- optional colour overlay (GUI display path only) ---
    if type == "disp":
        h, w = pred.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        # pred is already the band-Otsu binarised map; recompute band zones for display
        gt_band        = _dilate(gt,   EVAL_BAND_PX)
        gt_dilated     = _dilate(gt,   TOLERANCE_PX)
        pred_in_band   = np.logical_and(pred == 1, gt_band == 1).astype(np.uint8)
        pred_dilated   = _dilate(pred_in_band, TOLERANCE_PX)
        # GT evaluation band — faint grey background reference
        overlay[gt_band == 1]                                            = (50,  50,  50)
        # FN: GT contour pixel not covered by any in-band prediction — blue
        overlay[(gt == 1) & (pred_dilated == 0)]                         = (200,  0,   0)
        # FP: in-band predicted pixel outside the GT tolerance zone — red
        overlay[(pred_in_band == 1) & (gt_dilated == 0)]                 = (0,    0,  200)
        # TP: in-band predicted pixel within GT tolerance zone — green
        overlay[(pred_in_band == 1) & (gt_dilated == 1)]                 = (0,  200,   0)
        cv2.imwrite('iou.png', overlay)
 
    return {"iou": iou_score, "precision": precision, "recall": recall, "f1": f1}
 
 
# Map GUI label strings to the array-based processing functions
_PREPROCESS_FN = {
    "Gaussian Smoothing":     pre_processors.gaussSmooth_arr,
    "Median Filtering":       pre_processors.medianFilter_arr,
    "Bilateral Filtering":    pre_processors.bilateralFilter_arr,
    "Histogram Equalization": pre_processors.histogramEqual_arr,
    "CLAHE":                  pre_processors.CLAHE_arr,
}
_DEGRADE_FN = {
    "Gaussian Noise":         controlled_degradations.gaussNoise_arr,
    "Salt and Pepper Noise":  controlled_degradations.saltpepper_arr,
    "Blur":                   controlled_degradations.blur_arr,
    "Reduce Illumination":    controlled_degradations.debright_arr,
}
_EDGE_FN = {
    "Sobel":   edge_detectors.sobel_arr,
    "Prewitt": edge_detectors.prewitt_arr,
    "LoG":     edge_detectors.log_arr,
    "Canny":   edge_detectors.canny_arr,
}
 
 
def score_image(imgpath, preprocessor, degradation, edge_detector):
    """Run the full pipeline on a single image entirely in memory.
 
    Returns (metrics_dict, anno_path, skip_reason) where:
      - metrics_dict  : dict with iou/precision/recall/f1, or None on failure
      - anno_path     : the expected annotation mask path (for diagnostics)
      - skip_reason   : None on success, or a short string explaining the skip
    No temporary files are written.
    """
    stem = Path(imgpath).stem
 
    # Skip non-image files that may be present in the dataset directory
    if not stem or stem.startswith('.'):
        return None, '', 'non-image file'
 
    anno_path = f"annotated-masks/{stem}_segmentation.png"
    if not Path(anno_path).exists():
        return None, anno_path, 'no mask'
 
    if edge_detector not in _EDGE_FN:
        return None, anno_path, f'unknown detector: {edge_detector}'
 
    img = cv2.imread(imgpath)
    if img is None:
        return None, anno_path, 'image unreadable'
 
    if preprocessor and preprocessor != "None" and preprocessor in _PREPROCESS_FN:
        img = _PREPROCESS_FN[preprocessor](img)
 
    if degradation and degradation != "None" and degradation in _DEGRADE_FN:
        img = _DEGRADE_FN[degradation](img)
 
    edge_arr = _EDGE_FN[edge_detector](img)
 
    anno_arr = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
    if anno_arr is None:
        return None, anno_path, 'mask unreadable'
 
    metrics = iou(edge_arr, anno_arr, "calc")
    return metrics, anno_path, None
 
 
def _stat(values):
    """Return (mean, median, std, min, max) for a list of floats."""
    a = np.array(values)
    return float(np.mean(a)), float(np.median(a)), float(np.std(a)), float(np.min(a)), float(np.max(a))
 
 
def calculate(dataset, preprocessor, degradation, edge_detector,
              progress_callback=None):
    """Run the full pipeline over every image in dataset and return a results string.
 
    All processing is done in memory via the _arr functions — no temp files are
    written or read during batch evaluation.
 
    progress_callback(i, total) is called after each image if provided.
 
    Returns a multi-line summary string suitable for display in the GUI label.
    """
    ious, precisions, recalls, f1s = [], [], [], []
    skipped = 0
    total = len(dataset)
 
    skip_reasons = {}   # reason -> count, for diagnostic summary
 
    for i, imgpath in enumerate(dataset, 1):
        metrics, _, reason = score_image(imgpath, preprocessor, degradation, edge_detector)
        if metrics is None:
            skipped += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        else:
            ious.append(metrics["iou"])
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            f1s.append(metrics["f1"])
 
        if progress_callback is not None:
            progress_callback(i, total)
 
    if not ious:
        reason_lines = "\n".join(
            f"  {r or 'unknown'}: {c}" for r, c in sorted(skip_reasons.items()))
        return (
            "Dataset Results\n"
            "───────────────────────────────\n"
            f"Images processed : 0\n"
            f"Images skipped   : {skipped}\n"
            "Skip reasons:\n" + reason_lines
        )
 
    mi, mdi, si, ni, xi           = _stat(ious)
    mp, mdp, sp, np_, xp          = _stat(precisions)
    mr, mdr, sr, nr, xr           = _stat(recalls)
    mf, mdf, sf, nf, xf           = _stat(f1s)
 
    rating = "Good" if mf >= 0.75 else "Fair" if mf >= 0.50 else "Poor"
 
    skip_detail = ("  " + ", ".join(f"{r}: {c}" for r, c in sorted(skip_reasons.items()))
                   if skip_reasons else "  none")

    # print the calculated results for csv importing
    print(f"Making CSV import for: {preprocessor or 'None'}, {degradation or 'None'}, {edge_detector}")
    print("Format: Metric,Mean,Median,Std,Min,Max")
    csv_list = [
        ["IoU",f"{mi:>7.4f}",f"{mdi:>7.4f}",f"{si:>7.4f}",f"{ni:>7.4f}",f"{xi:>7.4f}"],
        ["Precision",f"{mp:>7.4f}",f"{mdp:>7.4f}",f"{sp:>7.4f}",f"{np_:>7.4f}",f"{xp:>7.4f}"],
        ["Recall",f"{mr:>7.4f}",f"{mdr:>7.4f}",f"{sr:>7.4f}",f"{nr:>7.4f}",f"{xr:>7.4f}"],
        ["F1",f"{mf:>7.4f}",f"{mdf:>7.4f}",f"{sf:>7.4f}",f"{nf:>7.4f}",f"{xf:>7.4f}"]
    ]

    csv_conversion = io.StringIO()
    writer = csv.writer(csv_conversion)
    writer.writerows(csv_list)
    csv_string= csv_conversion.getvalue()
    print(csv_string)

    return (
        f"Dataset Results\n"
        f"───────────────────────────────\n"
        f"Preprocessor  : {preprocessor or 'None'}\n"
        f"Degradation   : {degradation or 'None'}\n"
        f"Edge Detector : {edge_detector}\n"
        f"───────────────────────────────\n"
        f"Images scored : {len(ious)}\n"
        f"Images skipped: {skipped}\n"
        f"{skip_detail}\n"
        f"───────────────────────────────\n"
        f"{'Metric':<10} {'Mean':>7} {'Median':>7} {'Std':>7} {'Min':>7} {'Max':>7}\n"
        f"{'IoU':<10} {mi:>7.4f} {mdi:>7.4f} {si:>7.4f} {ni:>7.4f} {xi:>7.4f}\n"
        f"{'Precision':<10} {mp:>7.4f} {mdp:>7.4f} {sp:>7.4f} {np_:>7.4f} {xp:>7.4f}\n"
        f"{'Recall':<10} {mr:>7.4f} {mdr:>7.4f} {sr:>7.4f} {nr:>7.4f} {xr:>7.4f}\n"
        f"{'F1':<10} {mf:>7.4f} {mdf:>7.4f} {sf:>7.4f} {nf:>7.4f} {xf:>7.4f}\n"
        f"───────────────────────────────\n"
        f"Performance   : {rating}"
    )
 
