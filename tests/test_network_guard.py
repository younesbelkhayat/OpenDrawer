"""Meta-tests: prove the session network guard blocks downloads/uploads."""

from __future__ import annotations

import socket

import pytest

from lib.model_registry import ModelRegistry


class TestGuardBlocksOutbound:
    def test_raw_socket_connect_is_blocked(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(Exception) as exc:
            sock.connect(("huggingface.co", 443))
        assert "Blocked a network connection" in str(exc.value)

    def test_create_connection_is_blocked(self):
        with pytest.raises(Exception) as exc:
            socket.create_connection(("huggingface.co", 443), timeout=5)
        assert "Blocked a network connection" in str(exc.value)

    def test_loopback_still_permitted(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            socket.create_connection(("127.0.0.1", port), timeout=5).close()
        finally:
            server.close()

    def test_model_download_cannot_happen(self, tmp_path):
        registry = ModelRegistry(models_dir=tmp_path)
        with pytest.raises(Exception):
            registry.download("yunet_face")
        assert not registry.is_downloaded("yunet_face")


class TestLiveApiMarkerIsSkipped:
    @pytest.mark.live_api
    def test_this_should_never_run_by_default(self):
        raise AssertionError("A @live_api test ran without OPENDRAWER_ALLOW_NETWORK=1")
