import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["BENCH_WORKSPACE"])

CONFIG = Path(os.environ["BENCH_WORKSPACE"]) / "config.json"


def _load():
    return json.loads(CONFIG.read_text())


def test_version_bumped_to_2():
    assert _load()["version"] == 2


def test_debug_is_false():
    assert _load()["debug"] is False


def test_allowed_hosts_keeps_localhost():
    assert "localhost" in _load()["allowedHosts"]


def test_allowed_hosts_gains_api_example_com():
    assert "api.example.com" in _load()["allowedHosts"]


def test_server_block_untouched():
    server = _load()["server"]
    assert server["host"] == "localhost"
    assert server["port"] == 8080
