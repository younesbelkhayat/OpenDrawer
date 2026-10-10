"""Backlot board state for coloring projects."""

from __future__ import annotations

from backlot.state import list_projects, load_board_state


def test_page_board_state(e2e_page):
    projects_dir, pid = e2e_page
    s = load_board_state(projects_dir / pid)
    assert [st["name"] for st in s["stages"]] == ["intake", "cartoon", "line_art", "regions", "page"]
    assert all(st["status"] == "completed" for st in s["stages"])
    steps = {st["stage"]: st for st in s["steps"]}
    assert steps["page"]["images"][0]["path"] == "output/page.png"
    assert sum(1 for i in steps["cartoon"]["images"] if i.get("selected")) == 1
    assert s["qa"]["status"] in ("pass", "warn")
    assert s["poster"] == "output/page.png"


def test_library(e2e_page):
    projects_dir, pid = e2e_page
    cards = {c["project_id"]: c for c in list_projects(projects_dir)}
    assert cards[pid]["difficulty"] == "medium"
    assert cards[pid]["completed_count"] == 5


def test_server_health():
    from fastapi.testclient import TestClient

    from backlot.server import create_app

    with TestClient(create_app()) as client:
        assert client.get("/api/health").json()["ok"] is True
