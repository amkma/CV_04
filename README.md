# CV_04 — Image Thresholding & Unsupervised Segmentation

A complete, from-scratch implementation of image thresholding and unsupervised
segmentation algorithms. Built with **C++ / pybind11** for the compute core and
**Django** for the web interface. OpenCV is used **only** for `cv::Mat` storage
and image I/O — all algorithmic logic is implemented from scratch.

---

## Algorithms

### Thresholding (grayscale binarization)

| Method | Description |
|--------|-------------|
| **Optimal (Iterative)** | Initialize T as global mean, iteratively split pixels into two groups and update T = (μ₁+μ₂)/2 until convergence |
| **Otsu** | Exhaustive search for the threshold t that maximises between-class variance σ²_B = ω₀·ω₁·(μ₀ − μ₁)² |
| **Spectral (Multi-level)** | Recursive Otsu on sub-ranges of the histogram to produce N−1 thresholds for N classes |
| **Local (Adaptive)** | Divide image into B×B blocks, apply Otsu independently to each block — handles uneven illumination |

### Segmentation (colour or grayscale)

| Method | Description |
|--------|-------------|
| **K-Means** | k-means++ initialisation, iterative assignment + centroid update in pixel feature space (1D gray or 3D BGR) |
| **Region Growing** | BFS expansion from user-placed seed points; similarity = \|pixel − region_mean\| < threshold |
| **Agglomerative** | Bottom-up merging of 8×8 super-pixel blocks by average-linkage colour distance until target cluster count |
| **Mean Shift** | Iterative mode-seeking in joint (x, y, colour) space with flat kernels; converged modes merged |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Web Interface (Browser)                     │
│  ┌─────────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │ 2 Main Tabs │  │ Control Panel │  │ Results + Canvas │  │
│  │ Threshold   │  │ Method select │  │ Base64 images    │  │
│  │ Segment     │  │ Params sliders│  │ Histogram canvas │  │
│  └─────┬───────┘  └───────┬───────┘  └────────▲─────────┘  │
└────────│──────────────────│────────────────────│─────────────┘
         │                  │ (FormData)         │ (JSON+base64)
┌────────▼──────────────────▼────────────────────┴─────────────┐
│              Django Backend (views.py)                         │
│  - Upload handler, 8 algorithm endpoints, save endpoint       │
│  - All processing in RAM, base64 response                     │
└──────────────────────────┬───────────────────────────────────┘
                           │ (pybind11)
┌──────────────────────────▼───────────────────────────────────┐
│              C++ Core Engine (cv_core.pyd)                     │
│  cv_thresholding.h:                                           │
│  ├── Histogram computation                                    │
│  ├── Optimal thresholding (iterative)                         │
│  ├── Otsu thresholding (between-class variance)               │
│  ├── Spectral thresholding (recursive multi-level Otsu)       │
│  └── Local/adaptive thresholding (block-based Otsu)           │
│  cv_segmentation.h:                                           │
│  ├── K-means clustering (k-means++ init, gray+color)          │
│  ├── Region growing (BFS, similarity-based)                   │
│  ├── Agglomerative clustering (average linkage)               │
│  └── Mean shift (iterative mode-seeking)                      │
│  OpenCV ONLY for: cv::Mat, cv::imread, cv::imencode           │
└───────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- OpenCV 4.x (set `OpenCV_DIR` env var if not in default location)
- pybind11 (`pip install pybind11`)
- Django (`pip install django`)

### Build & Run

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build the C++ module
python build_cv_core.py build_ext --inplace

# Start the web server
python manage.py runserver
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## API Endpoints

| Endpoint | Method | Parameters |
|----------|--------|------------|
| `POST /api/upload/` | Upload image | `image` (file) |
| `POST /api/optimal-threshold/` | Optimal thresholding | `image_path` |
| `POST /api/otsu-threshold/` | Otsu thresholding | `image_path` |
| `POST /api/spectral-threshold/` | Spectral thresholding | `image_path`, `num_classes` |
| `POST /api/local-threshold/` | Local thresholding | `image_path`, `block_size` |
| `POST /api/kmeans/` | K-means segmentation | `image_path`, `k`, `max_iter` |
| `POST /api/region-growing/` | Region growing | `image_path`, `seeds` (JSON), `threshold` |
| `POST /api/agglomerative/` | Agglomerative clustering | `image_path`, `num_clusters` |
| `POST /api/mean-shift/` | Mean shift | `image_path`, `spatial_radius`, `color_radius` |
| `POST /api/save-result/` | Save result to disk | `image_b64`, `algorithm` |

All processing endpoints return `image_b64` (base64-encoded PNG) in JSON.

---

## File Structure

```
CV_04/
├── core/
│   ├── cv_core.cpp           # pybind11 module
│   ├── cv_utils.h            # Pixel access, grayscale, blur, drawing, base64
│   ├── cv_thresholding.h     # Optimal, Otsu, Spectral, Local thresholding
│   └── cv_segmentation.h     # K-means, Region Growing, Agglomerative, Mean Shift
├── cv_project/
│   ├── settings.py           # Django settings
│   ├── urls.py               # URL routing (10 endpoints)
│   └── wsgi.py
├── detector/
│   ├── views.py              # API endpoint handlers
│   └── templates/detector/
│       └── home.html         # Single-page application
├── static/
│   ├── css/style.css         # Dark glassmorphism theme
│   └── js/app.js             # Frontend logic
├── build_cv_core.py          # Setuptools build script
├── CMakeLists.txt            # CMake build config
└── requirements.txt
```
