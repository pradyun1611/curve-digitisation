#!/usr/bin/env python
"""
Reproducibility diagnostics script.

Prints system, Python, and dependency information to help debug
cross-machine behaviour differences in curve digitisation.

Usage:
    python scripts/diagnostics.py
"""

from __future__ import annotations

import os
import platform
import struct
import sys


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def system_info() -> None:
    _section("System / OS")
    print(f"  Platform      : {platform.platform()}")
    print(f"  Machine       : {platform.machine()}")
    print(f"  OS            : {platform.system()} {platform.release()}")
    print(f"  Python version: {sys.version}")
    print(f"  Architecture  : {struct.calcsize('P') * 8}-bit")
    print(f"  Byte order    : {sys.byteorder}")
    print(f"  Executable    : {sys.executable}")


def env_vars() -> None:
    _section("Relevant Environment Variables")
    for var in sorted([
        "SSL_CERT_FILE", "OPENAI_API_KEY", "AZURE_ENDPOINT",
        "AZURE_DEPLOYMENT_NAME", "OUTPUT_DIR",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "NUMPY_SEED",
        "CUDA_VISIBLE_DEVICES", "OPENCV_OPENCL_DEVICE",
    ]):
        val = os.environ.get(var)
        if var in ("OPENAI_API_KEY",) and val:
            val = val[:8] + "..." + val[-4:]  # redact
        print(f"  {var} = {val!r}")


def package_versions() -> None:
    _section("Package Versions (pip freeze style)")
    import importlib.metadata as _meta

    # Packages relevant to this project
    pkgs = [
        "numpy", "scipy", "scikit-learn", "scikit-image",
        "Pillow", "matplotlib", "opencv-python", "opencv-python-headless",
        "openai", "streamlit", "fastapi", "uvicorn", "httpx",
        "python-dotenv", "truststore",
    ]
    for name in pkgs:
        try:
            ver = _meta.version(name)
            print(f"  {name:30s} {ver}")
        except _meta.PackageNotFoundError:
            print(f"  {name:30s} ** NOT INSTALLED **")


def numpy_info() -> None:
    _section("NumPy Configuration")
    import numpy as np
    print(f"  version     : {np.__version__}")
    print(f"  float eps   : {np.finfo(np.float64).eps}")
    print(f"  int size    : {np.dtype(np.intp).itemsize * 8}-bit")
    try:
        info = np.show_config(mode="dicts")  # type: ignore[call-arg]
        if isinstance(info, dict):
            blas = info.get("Build Dependencies", {}).get("blas", {})
            if blas:
                print(f"  BLAS        : {blas.get('name', '?')} {blas.get('version', '?')}")
    except Exception:
        pass


def scipy_info() -> None:
    _section("SciPy Configuration")
    import scipy
    print(f"  version: {scipy.__version__}")


def opencv_info() -> None:
    _section("OpenCV (cv2)")
    try:
        import cv2
        print(f"  version       : {cv2.__version__}")
        print(f"  build info    :")
        for line in cv2.getBuildInformation().splitlines()[:30]:
            if line.strip():
                print(f"    {line.rstrip()}")
        print(f"  OpenCL enabled: {cv2.ocl.haveOpenCL()}")
        print(f"  OpenCL in use : {cv2.ocl.useOpenCL()}")
        print(f"  threads       : {cv2.getNumThreads()}")
    except ImportError:
        print("  ** cv2 NOT INSTALLED ** (optional – PIL is used as fallback)")


def pillow_info() -> None:
    _section("Pillow (PIL)")
    from PIL import Image, __version__ as pil_ver
    print(f"  version       : {pil_ver}")
    print(f"  SIMD support  : {getattr(Image.core, 'simd_support', 'unknown')}")


def threading_info() -> None:
    _section("Threading / Parallelism")
    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        print(f"  {var} = {os.environ.get(var, '(not set)')}")
    try:
        import cv2
        print(f"  cv2.getNumThreads() = {cv2.getNumThreads()}")
    except ImportError:
        pass


def quick_determinism_test() -> None:
    """Run a tiny synthetic extraction to verify determinism."""
    _section("Quick Determinism Smoke Test")
    import numpy as np
    from PIL import Image

    # Create a synthetic 200x150 image with a red and blue line
    img = Image.new("RGB", (200, 150), (255, 255, 255))
    pixels = img.load()
    for x in range(20, 180):
        pixels[x, 70] = (255, 0, 0)   # red line at y=70
        pixels[x, 71] = (255, 0, 0)
        pixels[x, 100] = (0, 0, 255)  # blue line at y=100
        pixels[x, 101] = (0, 0, 255)

    from core.image_processor import CurveDigitizer
    axis_info = {"xMin": 0, "xMax": 100, "yMin": 0, "yMax": 100}
    digi = CurveDigitizer(axis_info)

    red_px = digi.extract_color_pixels(img, "red")
    blue_px = digi.extract_color_pixels(img, "blue")
    is_gray = digi.is_grayscale_image(img)

    print(f"  Red pixels found  : {len(red_px)}")
    print(f"  Blue pixels found : {len(blue_px)}")
    print(f"  Is grayscale?     : {is_gray}")
    print(f"  Expected red ~320 : {'OK' if 280 <= len(red_px) <= 400 else 'FAIL'}")
    print(f"  Expected blue ~320: {'OK' if 280 <= len(blue_px) <= 400 else 'FAIL'}")
    print(f"  Expected gray=False: {'OK' if not is_gray else 'FAIL'}")


def main() -> None:
    print("Curve Digitisation – Reproducibility Diagnostics")
    print(f"Run at: {__import__('datetime').datetime.now().isoformat()}")

    system_info()
    env_vars()
    package_versions()
    numpy_info()
    scipy_info()
    opencv_info()
    pillow_info()
    threading_info()
    quick_determinism_test()

    print(f"\n{'=' * 60}")
    print("  Done. Share this output when reporting cross-machine issues.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # Ensure the project root is on sys.path so `core` is importable
    import pathlib
    _root = str(pathlib.Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    main()
