"""Session-wide test safety net.

**No test may open a network connection.** OpenDrawer promises that a run is
fully offline once the weights are on disk (children's photos never leave the
machine), and tests must never trigger a multi-gigabyte model download. This
blocks outbound sockets for the whole test session.

The guard is at the socket layer on purpose: `requests`, `huggingface_hub`,
`httpx` and raw `urllib` all bottom out in `socket.connect`.

Loopback is still allowed — the ComfyUI client and Backlot fixtures need it.

To write a test that genuinely needs the network (e.g. a real download):

    @pytest.mark.live_api
    def test_real_call():
        ...

Marked tests are **skipped by default** and only run with the env flag set:

    OPENDRAWER_ALLOW_NETWORK=1 pytest -m live_api

Limitation: this guards the pytest process. A test that shells out to a
subprocess is outside its reach — don't download from a subprocess in tests.
"""

from __future__ import annotations

import os
import socket

import pytest

_ALLOW_ENV_FLAG = "OPENDRAWER_ALLOW_NETWORK"

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_create_connection = socket.create_connection


class NetworkCallInTestError(RuntimeError):
    """Raised when a test tries to open a non-loopback connection."""


def _network_allowed() -> bool:
    return os.environ.get(_ALLOW_ENV_FLAG, "").strip().lower() in {"1", "true", "yes"}


def _is_loopback(address) -> bool:
    """True for loopback TCP/UDP targets and for AF_UNIX socket paths."""
    if isinstance(address, (str, bytes)):
        return True  # AF_UNIX / abstract socket — local by definition
    if not isinstance(address, (tuple, list)) or not address:
        return True  # unrecognised shape; let the real call decide
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")
    if not isinstance(host, str):
        return False
    host = host.strip("[]").lower()
    if host in _LOOPBACK_HOSTS:
        return True
    return host.startswith("127.")


def _blocked(address) -> NetworkCallInTestError:
    return NetworkCallInTestError(
        f"Blocked a network connection to {address!r} during a test.\n"
        f"\n"
        f"Tests must stay offline (no model downloads, no photo uploads).\n"
        f"Mock the transport instead, or mark the test @pytest.mark.live_api and run\n"
        f"it deliberately with {_ALLOW_ENV_FLAG}=1."
    )


@pytest.fixture(scope="session", autouse=True)
def _block_network():
    """Refuse non-loopback sockets for the entire session."""
    if _network_allowed():
        yield
        return

    def guarded_connect(self, address, *args, **kwargs):
        if not _is_loopback(address):
            raise _blocked(address)
        return _real_connect(self, address, *args, **kwargs)

    def guarded_connect_ex(self, address, *args, **kwargs):
        if not _is_loopback(address):
            raise _blocked(address)
        return _real_connect_ex(self, address, *args, **kwargs)

    def guarded_create_connection(address, *args, **kwargs):
        if not _is_loopback(address):
            raise _blocked(address)
        return _real_create_connection(address, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.create_connection = guarded_create_connection
    try:
        yield
    finally:
        socket.socket.connect = _real_connect
        socket.socket.connect_ex = _real_connect_ex
        socket.create_connection = _real_create_connection


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live_api: test needs the network (e.g. a real model download). Skipped unless "
        f"{_ALLOW_ENV_FLAG}=1 is set.",
    )


def pytest_collection_modifyitems(config, items):
    """Skip live_api tests unless the operator explicitly opted in."""
    if _network_allowed():
        return
    skip = pytest.mark.skip(
        reason=f"network test; set {_ALLOW_ENV_FLAG}=1 to run"
    )
    for item in items:
        if "live_api" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Shared fixtures: a synthetic photo and one end-to-end run (OpenCV backend,
# no GPU, no weights) reused by the pipeline, QA and Backlot tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def demo_photo(tmp_path_factory):
    from opendrawer.demo import make_demo_photo

    return make_demo_photo(tmp_path_factory.mktemp("photos") / "lea.jpg")


@pytest.fixture(scope="session")
def projects_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("projects")


@pytest.fixture(scope="session")
def e2e_page(demo_photo, projects_dir):
    """A completed coloring-page project: (projects_dir, project_id)."""
    from opendrawer.runner import AutoApprover, PageRun, RunSettings

    settings = RunSettings(difficulty="medium", backend="opencv", candidates=2, seed=1)
    run = PageRun.create(demo_photo, settings, AutoApprover(), project_id="e2e", projects_dir=projects_dir, log=lambda _: None)
    assert run.run() == "completed"
    return projects_dir, "e2e"
