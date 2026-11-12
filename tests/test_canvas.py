import pytest

from core.canvas import CanvasFullError, DrawingCanvas, Tool


def draw(canvas, points):
    canvas.begin_stroke(points[0])
    for p in points[1:]:
        canvas.add_point(p)
    return canvas.end_stroke()


def test_strokes_persist_in_normalized_coordinates():
    c = DrawingCanvas()
    stroke = draw(c, [(0.1, 0.2), (0.3, 0.4), (1.5, -0.2)])
    assert stroke.points == ((0.1, 0.2), (0.3, 0.4), (1.0, 0.0))  # Clamped to [0, 1].
    assert c.strokes == (stroke,)


def test_active_stroke_is_not_in_history_until_completed():
    c = DrawingCanvas()
    c.begin_stroke((0.1, 0.1))
    c.add_point((0.2, 0.2))
    assert c.strokes == ()
    assert c.active_stroke is not None
    c.end_stroke()
    assert len(c.strokes) == 1 and c.active_stroke is None


def test_single_point_stroke_is_a_dot():
    c = DrawingCanvas()
    assert len(draw(c, [(0.5, 0.5)]).points) == 1


def test_points_closer_than_min_distance_are_skipped():
    c = DrawingCanvas(min_point_distance=0.01)
    stroke = draw(c, [(0.5, 0.5), (0.501, 0.5), (0.502, 0.5), (0.52, 0.5)])
    assert len(stroke.points) == 2


# --- 9. eraser and undo ------------------------------------------------------------

def test_eraser_strokes_are_flagged_and_wider():
    c = DrawingCanvas(thickness=0.01, eraser_factor=3.0)
    c.set_tool(Tool.ERASER)
    stroke = draw(c, [(0.1, 0.1), (0.2, 0.2)])
    assert stroke.erase is True
    assert stroke.thickness == pytest.approx(0.03)


def test_undo_removes_last_completed_stroke():
    c = DrawingCanvas()
    a = draw(c, [(0.1, 0.1), (0.2, 0.2)])
    draw(c, [(0.5, 0.5), (0.6, 0.6)])
    assert c.undo() is True
    assert c.strokes == (a,)


def test_undo_during_stroke_drops_that_stroke():
    c = DrawingCanvas()
    a = draw(c, [(0.1, 0.1), (0.2, 0.2)])
    c.begin_stroke((0.5, 0.5))
    c.add_point((0.6, 0.6))
    c.undo()
    assert c.strokes == (a,) and not c.is_drawing


def test_undo_eraser_restores_erased_area():
    c = DrawingCanvas()
    draw(c, [(0.1, 0.1), (0.9, 0.9)])
    c.set_tool(Tool.ERASER)
    draw(c, [(0.4, 0.4), (0.6, 0.6)])
    assert len(c.strokes) == 2
    c.undo()
    assert len(c.strokes) == 1 and not c.strokes[0].erase


def test_undo_with_empty_history():
    assert DrawingCanvas().undo() is False


def test_undo_history_is_bounded():
    c = DrawingCanvas(undo_limit=3)
    for i in range(6):
        draw(c, [(0.1 * i, 0.5)])
    assert sum(c.undo() for _ in range(10)) == 3
    assert len(c.strokes) == 3  # Older strokes stay; they just can't be undone.


def test_clear_is_undoable():
    c = DrawingCanvas()
    draw(c, [(0.1, 0.1), (0.2, 0.2)])
    draw(c, [(0.3, 0.3), (0.4, 0.4)])
    assert c.clear() is True and c.is_empty
    c.undo()
    assert len(c.strokes) == 2


def test_clear_empty_canvas_does_nothing():
    c = DrawingCanvas()
    assert c.clear() is False
    assert c.undo() is False


def test_stroke_cap_is_enforced():
    c = DrawingCanvas(max_strokes=2)
    draw(c, [(0.1, 0.1)])
    draw(c, [(0.2, 0.2)])
    with pytest.raises(CanvasFullError):
        c.begin_stroke((0.3, 0.3))


def test_long_stroke_is_split_without_gap():
    c = DrawingCanvas(max_points_per_stroke=5, min_point_distance=0.0)
    c.begin_stroke((0.0, 0.5))
    for i in range(1, 12):
        c.add_point((i * 0.05, 0.5))
    c.end_stroke()
    strokes = c.strokes
    assert len(strokes) >= 2
    assert all(len(s.points) <= 5 for s in strokes)
    for a, b in zip(strokes, strokes[1:]):
        assert a.points[-1] == b.points[0]  # Continuous.


def test_version_changes_only_on_committed_changes():
    c = DrawingCanvas()
    v0 = c.version
    c.begin_stroke((0.1, 0.1))
    c.add_point((0.2, 0.2))
    assert c.version == v0
    c.end_stroke()
    assert c.version == v0 + 1
