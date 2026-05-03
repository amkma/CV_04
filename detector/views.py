"""Views for the detector app — CV_04.

Image Thresholding & Unsupervised Segmentation.
All results returned as base64-encoded images in JSON (no intermediate disk writes).
"""

import json
import os
import uuid
import time

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# ── Make sure OpenCV DLLs are findable ──
_opencv_candidates = [
    r"C:\Program Files\opencv\build\bin",
    r"C:\Program Files\opencv\build\x64\vc16\bin",
    r"C:\Program Files\opencv\build\x64\vc17\bin",
    r"C:\opencv\build\bin",
    r"C:\opencv\build\x64\vc16\bin",
    r"C:\opencv\build\x64\vc17\bin",
]
for _opencv_bin in _opencv_candidates:
    if os.path.isdir(_opencv_bin):
        os.add_dll_directory(_opencv_bin)
        if _opencv_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _opencv_bin + ";" + os.environ.get("PATH", "")

import cv_core


# ── Helpers ────────────────────────────────────────────────────────

def _save_upload(f):
    """Save uploaded file, return absolute path."""
    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(f.name)[1].lower() or ".png"
    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(upload_dir, name)
    with open(path, "wb") as dest:
        for chunk in f.chunks():
            dest.write(chunk)
    return path


def _get_image_path(request):
    """Extract and validate image_path from POST data."""
    img_path = request.POST.get("image_path", "")
    if not img_path or not os.path.isfile(img_path):
        return None
    return img_path


def _results_dir():
    d = os.path.join(settings.MEDIA_ROOT, "results")
    os.makedirs(d, exist_ok=True)
    return d


# ── Single page ───────────────────────────────────────────────────

def home(request):
    return render(request, "detector/home.html")


# ── API: upload image ─────────────────────────────────────────────

@csrf_exempt
def api_upload(request):
    if request.method != "POST" or "image" not in request.FILES:
        return JsonResponse({"error": "POST with image required"}, status=400)
    path = _save_upload(request.FILES["image"])
    return JsonResponse({"path": path})


# ═══════════════════════════════════════════════════════════════════
#  THRESHOLDING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@csrf_exempt
def api_optimal_threshold(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    img_path = _get_image_path(request)
    if not img_path:
        return JsonResponse({"error": "Invalid image path"}, status=400)

    try:
        result = cv_core.optimal_threshold(img_path)
        return JsonResponse({
            "image_b64": result["image_b64"],
            "threshold": result["threshold"],
            "iterations": result["iterations"],
            "time_ms": round(result["time_ms"], 2),
            "histogram": result["histogram"],
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_otsu_threshold(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    img_path = _get_image_path(request)
    if not img_path:
        return JsonResponse({"error": "Invalid image path"}, status=400)

    try:
        result = cv_core.otsu_threshold(img_path)
        return JsonResponse({
            "image_b64": result["image_b64"],
            "threshold": result["threshold"],
            "time_ms": round(result["time_ms"], 2),
            "histogram": result["histogram"],
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_spectral_threshold(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    img_path = _get_image_path(request)
    if not img_path:
        return JsonResponse({"error": "Invalid image path"}, status=400)

    num_classes = int(request.POST.get("num_classes", 3))

    try:
        result = cv_core.spectral_threshold(img_path, num_classes)
        return JsonResponse({
            "image_b64": result["image_b64"],
            "num_classes": result["num_classes"],
            "thresholds": list(result["thresholds"]),
            "time_ms": round(result["time_ms"], 2),
            "histogram": result["histogram"],
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_local_threshold(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    img_path = _get_image_path(request)
    if not img_path:
        return JsonResponse({"error": "Invalid image path"}, status=400)

    block_size = int(request.POST.get("block_size", 32))

    try:
        result = cv_core.local_threshold(img_path, block_size)
        return JsonResponse({
            "image_b64": result["image_b64"],
            "block_size": result["block_size"],
            "num_blocks": result["num_blocks"],
            "time_ms": round(result["time_ms"], 2),
            "histogram": result["histogram"],
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════════════
#  SEGMENTATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@csrf_exempt
def api_kmeans(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    img_path = _get_image_path(request)
    if not img_path:
        return JsonResponse({"error": "Invalid image path"}, status=400)

    k = int(request.POST.get("k", 4))
    max_iter = int(request.POST.get("max_iter", 50))

    try:
        result = cv_core.kmeans_segment(img_path, k, max_iter)
        return JsonResponse({
            "image_b64": result["image_b64"],
            "num_clusters": result["num_clusters"],
            "iterations": result["iterations"],
            "time_ms": round(result["time_ms"], 2),
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_region_growing(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    img_path = _get_image_path(request)
    if not img_path:
        return JsonResponse({"error": "Invalid image path"}, status=400)

    threshold = float(request.POST.get("threshold", 15.0))

    # Parse seeds from JSON string: [[y, x], [y, x], ...]
    seeds_str = request.POST.get("seeds", "[]")
    try:
        seeds_raw = json.loads(seeds_str)
        seeds = [(int(s[0]), int(s[1])) for s in seeds_raw]
    except (json.JSONDecodeError, IndexError, TypeError):
        return JsonResponse({"error": "Invalid seeds format"}, status=400)

    if not seeds:
        return JsonResponse({"error": "At least one seed point is required"}, status=400)

    try:
        result = cv_core.region_growing(img_path, seeds, threshold)
        return JsonResponse({
            "image_b64": result["image_b64"],
            "num_regions": result["num_regions"],
            "time_ms": round(result["time_ms"], 2),
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_agglomerative(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    img_path = _get_image_path(request)
    if not img_path:
        return JsonResponse({"error": "Invalid image path"}, status=400)

    num_clusters = int(request.POST.get("num_clusters", 4))

    try:
        result = cv_core.agglomerative_segment(img_path, num_clusters)
        return JsonResponse({
            "image_b64": result["image_b64"],
            "num_clusters": result["num_clusters"],
            "iterations": result["iterations"],
            "time_ms": round(result["time_ms"], 2),
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_mean_shift(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    img_path = _get_image_path(request)
    if not img_path:
        return JsonResponse({"error": "Invalid image path"}, status=400)

    spatial_radius = float(request.POST.get("spatial_radius", 20.0))
    color_radius = float(request.POST.get("color_radius", 30.0))

    try:
        result = cv_core.mean_shift_segment(img_path, spatial_radius, color_radius)
        return JsonResponse({
            "image_b64": result["image_b64"],
            "num_clusters": result["num_clusters"],
            "time_ms": round(result["time_ms"], 2),
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════════════
#  SAVE ENDPOINT — user-triggered disk write
# ═══════════════════════════════════════════════════════════════════

@csrf_exempt
def api_save_result(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    b64_data = request.POST.get("image_b64", "")
    algorithm = request.POST.get("algorithm", "result")

    if not b64_data:
        return JsonResponse({"error": "No image data provided"}, status=400)

    # Generate output filename
    timestamp = int(time.time())
    filename = f"CV04_{algorithm}_{timestamp}.png"
    out_path = os.path.join(_results_dir(), filename)

    try:
        result = cv_core.save_result(out_path, b64_data)
        return JsonResponse({
            "saved_path": result["saved_path"],
            "filename": filename,
            "size_bytes": result["size_bytes"],
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
