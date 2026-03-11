"""Quick validation of the coordinate mapping and fitting fixes."""
import numpy as np
import sys

def test_bic_linear():
    from core.bw_fit import _fit_poly_bic
    xs = np.linspace(0.1, 0.9, 100)
    ys = 2.5 * xs + 0.3 + np.random.normal(0, 0.01, 100)
    r = _fit_poly_bic(xs, ys, min_degree=1, max_degree=4)
    assert r["degree"] == 1, f"Expected degree 1, got {r['degree']}"
    assert r["r_squared"] > 0.99, f"R² too low: {r['r_squared']}"
    print(f"  BIC linear: degree={r['degree']} R²={r['r_squared']:.6f} PASS")

def test_bic_quadratic():
    from core.bw_fit import _fit_poly_bic
    xs = np.linspace(0.1, 0.9, 100)
    ys = 3 * xs**2 - 2 * xs + 1 + np.random.normal(0, 0.01, 100)
    r = _fit_poly_bic(xs, ys, min_degree=1, max_degree=4)
    assert r["degree"] == 2, f"Expected degree 2, got {r['degree']}"
    print(f"  BIC quadratic: degree={r['degree']} R²={r['r_squared']:.6f} PASS")

def test_smooth_linear():
    from core.bw_pipeline import smooth_curve
    xs = np.linspace(0, 1, 80)
    ys = 3.0 * xs + 1.0 + np.random.normal(0, 0.005, 80)
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
    smoothed = smooth_curve(pts)
    arr = np.array(smoothed)
    coeffs = np.polyfit(arr[:, 0], arr[:, 1], 1)
    pred = np.polyval(coeffs, arr[:, 0])
    ss_res = np.sum((arr[:, 1] - pred) ** 2)
    ss_tot = np.sum((arr[:, 1] - np.mean(arr[:, 1])) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.999, f"Smoothing distorted linear data: R²={r2}"
    print(f"  Smooth linear: R²={r2:.6f} PASS")

def test_fence_post_roundtrip():
    from core.reconstruction import _data_to_pixel_simple
    from core.image_processor import CurveDigitizer
    
    pa_left, pa_top, pa_right, pa_bottom = 100, 50, 600, 450
    img_w, img_h = 700, 500
    axis_info = {"xMin": 0.0, "xMax": 10.0, "yMin": 0.0, "yMax": 100.0}
    plot_area = [pa_left, pa_top, pa_right, pa_bottom]
    
    p_width = max(pa_right - pa_left - 1, 1)
    p_height = max(pa_bottom - pa_top - 1, 1)
    
    # Forward mapping (pixel -> normalized -> data)
    test_pixels = [
        (100, 450),  # bottom-left corner -> (0, 0)
        (600, 50),   # top-right corner -> (10, 100)
        (350, 250),  # center
    ]
    for px_x, px_y in test_pixels:
        norm_x = (px_x - pa_left) / p_width
        norm_y = 1.0 - (px_y - pa_top) / p_height
        data_x = norm_x * 10.0
        data_y = norm_y * 100.0
        
        # Inverse: data -> pixel via _data_to_pixel_simple
        inv = _data_to_pixel_simple(
            [(data_x, data_y)], axis_info, img_w, img_h, plot_area
        )
        inv_x, inv_y = inv[0]
        dx = abs(inv_x - px_x)
        dy = abs(inv_y - px_y)
        assert dx <= 1 and dy <= 1, (
            f"Round-trip failed for ({px_x},{px_y}): data=({data_x:.2f},{data_y:.2f}) "
            f"got ({inv_x},{inv_y}) error=({dx},{dy})"
        )
    print(f"  Fence-post round-trip: {len(test_pixels)} points PASS")

if __name__ == "__main__":
    print("Testing fixes...")
    test_bic_linear()
    test_bic_quadratic()
    test_smooth_linear()
    test_fence_post_roundtrip()
    print("All tests PASSED")
