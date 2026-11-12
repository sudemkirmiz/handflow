import datetime as dt

import numpy as np
import pytest
from PIL import Image

from core.canvas import DrawingCanvas, Stroke, Tool
from core.canvas_render import (
    LayerCache, composite, export_image, export_size, render_strokes, save_png, unique_export_path,
)
from core.viewport import fit_viewport, landmark_to_canvas

PINK = (255, 143, 207)


# --- 6. mirroring and coordinate mapping -------------------------------------------------

def test_letterbox_wide_area():
    vp = fit_viewport(640, 480, 1000, 500)
    assert (vp.display_w, vp.display_h) == (666, 500)
    assert (vp.offset_x, vp.offset_y) == (167, 0)


def test_letterbox_tall_area():
    vp = fit_viewport(640, 480, 400, 900)
    assert (vp.display_w, vp.display_h) == (400, 300)
    assert (vp.offset_x, vp.offset_y) == (0, 300)


@pytest.mark.parametrize("frame", [(640, 480), (1280, 720), (480, 640)])
@pytest.mark.parametrize("area", [(900, 600), (500, 900), (1920, 1080)])
def test_display_mapping_round_trips(frame, area):
    vp = fit_viewport(*frame, *area)
    for u, v in [(0, 0), (1, 1), (0.25, 0.8), (0.5, 0.5)]:
        x, y = vp.to_display(u, v)
        assert 0 <= x <= area[0] and 0 <= y <= area[1]
        back = vp.to_normalized(x, y)
        assert back == pytest.approx((u, v), abs=1e-9)


def test_points_in_letterbox_margins_do_not_map():
    vp = fit_viewport(640, 480, 1000, 500)  # 167 px margins left and right.
    assert vp.to_normalized(50, 250) is None
    assert vp.to_normalized(990, 250) is None
    assert vp.to_normalized(500, 250) == pytest.approx((0.5, 0.5), abs=1e-3)


def test_empty_area_has_no_viewport():
    assert fit_viewport(640, 480, 0, 300) is None


def test_mirroring_rules():
    # HandFlow: detection input and display are the same mirrored frame -> identity.
    assert landmark_to_canvas(0.2, 0.7, True, True) == (0.2, 0.7)
    assert landmark_to_canvas(0.2, 0.7, False, False) == (0.2, 0.7)
    # Raw-frame detection shown mirrored (or vice versa) -> flip x.
    assert landmark_to_canvas(0.2, 0.7, False, True) == pytest.approx((0.8, 0.7))
    assert landmark_to_canvas(0.2, 0.7, True, False) == pytest.approx((0.8, 0.7))


# --- 7. resizing with persistent strokes ----------------------------------------------------

def alpha_at(layer, u, v):
    h, w = layer.shape[:2]
    return layer[int(round(v * (h - 1))), int(round(u * (w - 1))), 3]


@pytest.mark.parametrize("size", [(320, 240), (666, 500), (1600, 1200), (1280, 720)])
def test_strokes_render_at_same_relative_place_at_any_size(size):
    stroke = Stroke(((0.2, 0.5), (0.8, 0.5)), PINK, 0.02)
    layer = render_strokes([stroke], *size)
    assert alpha_at(layer, 0.5, 0.5) == 255  # On the line.
    assert alpha_at(layer, 0.2, 0.5) == 255
    assert alpha_at(layer, 0.5, 0.3) == 0  # Well off the line.
    # Thickness scales with height: ~2% of the height (OpenCV's anti-aliased
    # thick lines come out up to ~2 px wider than the nominal thickness).
    column = layer[:, size[0] // 2, 3]
    assert 0.02 * size[1] - 1 <= int((column > 127).sum()) <= 0.02 * size[1] + 3


def test_resizing_does_not_change_stored_strokes():
    c = DrawingCanvas()
    c.begin_stroke((0.3, 0.3))
    c.add_point((0.6, 0.6))
    c.end_stroke()
    before = c.strokes
    cache = LayerCache()
    for size in [(400, 300), (1000, 750), (400, 300)]:
        cache.layer_for(c, *size)
    assert c.strokes == before


def test_layer_cache_rerenders_only_when_needed():
    c = DrawingCanvas()
    c.begin_stroke((0.3, 0.3))
    c.end_stroke()
    cache = LayerCache()
    cache.layer_for(c, 400, 300)
    cache.layer_for(c, 400, 300)
    assert cache.renders == 1
    cache.layer_for(c, 500, 375)
    assert cache.renders == 2
    c.begin_stroke((0.5, 0.5))  # Active stroke: drawn on a copy, no full re-render.
    cache.layer_for(c, 500, 375)
    assert cache.renders == 2
    c.end_stroke()
    cache.layer_for(c, 500, 375)
    assert cache.renders == 3


# --- 9. eraser rendering / compositing ----------------------------------------------------------

def test_eraser_clears_pixels():
    brush = Stroke(((0.1, 0.5), (0.9, 0.5)), PINK, 0.05)
    eraser = Stroke(((0.5, 0.3), (0.5, 0.7)), (0, 0, 0), 0.1, erase=True)
    layer = render_strokes([brush, eraser], 400, 300)
    assert alpha_at(layer, 0.2, 0.5) == 255
    assert alpha_at(layer, 0.5, 0.5) == 0


def test_composite_blends_over_frame():
    frame = np.full((10, 10, 3), 100, np.uint8)
    layer = np.zeros((10, 10, 4), np.uint8)
    layer[0, 0] = (255, 143, 207, 255)  # Opaque.
    layer[1, 1] = (64, 0, 0, 128)  # ~50% premultiplied red-ish.
    out = composite(frame, layer)
    assert tuple(out[0, 0]) == PINK
    assert tuple(out[5, 5]) == (100, 100, 100)
    assert np.abs(out[1, 1].astype(int) - (114, 50, 50)).max() <= 1  # 100*(127/255) + layer, rounded


# --- 11. PNG export ------------------------------------------------------------------------------

def make_canvas():
    c = DrawingCanvas(thickness=0.03)
    c.set_color(PINK)
    c.begin_stroke((0.2, 0.5))
    c.add_point((0.8, 0.5))
    c.end_stroke()
    return c


def test_export_transparent_png_is_valid(tmp_path):
    path = save_png(make_canvas(), tmp_path / "out", aspect=(640, 480), long_side=800)
    assert path.parent == tmp_path / "out" and path.suffix == ".png"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        assert img.format == "PNG" and img.mode == "RGBA" and img.size == (800, 600)
        assert img.getpixel((400, 300)) == (*PINK, 255)  # Stroke, straight (not premultiplied) color.
        assert img.getpixel((400, 50))[3] == 0  # Transparent elsewhere.


def test_export_with_solid_background(tmp_path):
    path = save_png(make_canvas(), tmp_path, aspect=(1280, 720), long_side=640, background=(255, 255, 255))
    with Image.open(path) as img:
        assert img.mode == "RGB" and img.size == (640, 360)
        assert img.getpixel((10, 10)) == (255, 255, 255)
        assert img.getpixel((320, 180)) == PINK


def test_export_eraser_reveals_background():
    c = make_canvas()
    c.set_tool(Tool.ERASER)
    c.begin_stroke((0.5, 0.3))
    c.add_point((0.5, 0.7))
    c.end_stroke()
    img = export_image(c.strokes, (400, 300), background=(17, 19, 24))
    assert img.getpixel((200, 150)) == (17, 19, 24)


def test_export_antialiased_edges_are_not_darkened():
    img = export_image([Stroke(((0.2, 0.5), (0.8, 0.5)), PINK, 0.05)], (400, 300))
    rgba = np.asarray(img)
    edge = rgba[(rgba[..., 3] > 64) & (rgba[..., 3] < 235)]
    assert len(edge) > 0
    # Un-premultiplied edge colors stay close to the brush color, not to black.
    # (Below ~25 % alpha, 8-bit quantization allows larger but invisible errors.)
    assert np.abs(edge[:, :3].astype(int) - np.array(PINK)).max() <= 8


def test_export_never_overwrites(tmp_path):
    now = dt.datetime(2026, 10, 31, 12, 0, 0)
    first = unique_export_path(tmp_path, now)
    first.write_bytes(b"x")
    second = unique_export_path(tmp_path, now)
    assert first != second and not second.exists()


def test_export_size_keeps_aspect():
    assert export_size(640, 480, 1600) == (1600, 1200)
    assert export_size(1280, 720, 1600) == (1600, 900)
    assert export_size(480, 640, 1600) == (1200, 1600)


# --- incremental rendering of the stroke being drawn -------------------------------------------------

def _drawing_canvas():
    c = DrawingCanvas(thickness=0.02, min_point_distance=0.0)
    c.set_color(PINK)
    c.begin_stroke((0.1, 0.5))
    return c


def test_incremental_active_stroke_matches_full_render():
    import math as _m

    c = _drawing_canvas()
    cache = LayerCache()
    for k in range(1, 300):
        c.add_point((0.1 + 0.8 * k / 300, 0.5 + 0.25 * _m.sin(k / 15)))
        incremental = cache.layer_for(c, 640, 480)
    full = render_strokes([c.active_stroke], 640, 480)
    inc_a, full_a = incremental[..., 3].astype(int), full[..., 3].astype(int)
    assert cache.active_rebuilds == 1  # Built once, then only extended.
    # The line body is identical; only anti-aliased edge pixels at segment joints
    # are blended twice, which makes them slightly denser, never lighter.
    assert np.all(inc_a[full_a == 255] == 255)
    assert np.all(inc_a >= full_a)
    edge = inc_a - full_a
    assert np.percentile(edge[edge > 0], 95) <= 40 and edge.max() <= 64
    # Nothing outside the line is touched.
    assert np.all(inc_a[full_a == 0] == 0) or (inc_a[full_a == 0] > 0).sum() < 0.002 * full_a.size
    # Once the stroke is committed it is rendered in full again: saved drawings
    # and PNG exports are not affected by the live shortcut.
    c.end_stroke()
    committed = cache.layer_for(c, 640, 480)
    assert np.array_equal(committed, render_strokes(c.strokes, 640, 480))


def test_frame_cost_does_not_grow_with_stroke_length():
    import time as _t

    c = _drawing_canvas()
    cache = LayerCache()
    timings = []
    for k in range(1, 3001):
        c.add_point((0.1 + 0.8 * (k % 300) / 300, 0.2 + 0.6 * k / 3000))
        start = _t.perf_counter()
        cache.layer_for(c, 900, 675)
        timings.append(_t.perf_counter() - start)
    early, late = np.median(timings[100:400]), np.median(timings[-300:])
    assert late < early * 2 + 0.0005  # Was linear in the number of points before.


def test_active_layer_resets_on_new_stroke_size_change_and_undo():
    c = _drawing_canvas()
    cache = LayerCache()
    c.add_point((0.5, 0.5))
    cache.layer_for(c, 640, 480)
    cache.layer_for(c, 800, 600)  # Resized: rebuilt at the new size.
    assert cache.active_rebuilds == 2
    c.end_stroke()
    c.begin_stroke((0.2, 0.8))
    c.add_point((0.3, 0.8))
    layer = cache.layer_for(c, 800, 600)
    assert cache.active_rebuilds == 3
    assert alpha_at(layer, 0.25, 0.8) == 255 and alpha_at(layer, 0.35, 0.5) == 255  # Both strokes present.
    c.undo()  # Ends + removes the active stroke.
    layer = cache.layer_for(c, 800, 600)
    assert alpha_at(layer, 0.25, 0.8) == 0
