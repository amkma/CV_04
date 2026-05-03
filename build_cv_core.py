"""
Build script for the cv_core C++ extension module (CV_04).

Usage:
    python build_cv_core.py build_ext --inplace
"""

import os
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# ── Locate OpenCV ──
opencv_dir = os.environ.get("OpenCV_DIR", r"C:\Program Files\opencv\build")

include_dirs = [
    os.path.join(opencv_dir, "include"),
    os.path.join(os.path.dirname(__file__), "core"),  # our header files
]
library_dirs = []
libraries    = []

if sys.platform == "win32":
    for sub in ["x64/vc16/lib", "x64/vc17/lib", "x64/vc15/lib", "lib"]:
        candidate = os.path.join(opencv_dir, sub)
        if os.path.isdir(candidate):
            library_dirs.append(candidate)
            break

    found = False
    if library_dirs:
        import glob
        libs = glob.glob(os.path.join(library_dirs[0], "opencv_world*.lib"))
        for lib in sorted(libs):
            base = os.path.splitext(os.path.basename(lib))[0]
            if not base.endswith("d"):
                libraries = [base]
                found = True
                break
    if not found:
        libraries = ["opencv_world4120"]
else:
    import subprocess
    try:
        cflags  = subprocess.check_output(["pkg-config", "--cflags", "opencv4"], text=True).strip().split()
        ldflags = subprocess.check_output(["pkg-config", "--libs",   "opencv4"], text=True).strip().split()
        for f in cflags:
            if f.startswith("-I"): include_dirs.append(f[2:])
        for f in ldflags:
            if f.startswith("-L"): library_dirs.append(f[2:])
            elif f.startswith("-l"): libraries.append(f[2:])
    except Exception:
        libraries = ["opencv_core", "opencv_imgproc", "opencv_imgcodecs", "opencv_highgui"]

ext = Pybind11Extension(
    "cv_core",
    [os.path.join("core", "cv_core.cpp")],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    cxx_std=17,
)

setup(
    name="cv_core",
    version="0.4.0",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
