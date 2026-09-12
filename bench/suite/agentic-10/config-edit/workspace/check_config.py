#!/usr/bin/env python3
"""Quick checker for config.json. Run: python check_config.py"""
import json
import sys
from pathlib import Path


def check(cfg):
    problems = []
    if cfg.get("version") != 2:
        problems.append("version should be 2, found %r" % cfg.get("version"))
    if cfg.get("debug") is not False:
        problems.append("debug should be false, found %r" % cfg.get("debug"))
    hosts = cfg.get("allowedHosts")
    if not isinstance(hosts, list):
        problems.append("allowedHosts should be a list of host names")
    else:
        if "localhost" not in hosts:
            problems.append("allowedHosts is missing 'localhost'")
        if "api.example.com" not in hosts:
            problems.append("allowedHosts is missing 'api.example.com'")
    return problems


def main():
    cfg = json.loads(Path(__file__).with_name("config.json").read_text())
    problems = check(cfg)
    if problems:
        for problem in problems:
            print("FAIL: " + problem)
        return 1
    print("config.json looks good")
    return 0


if __name__ == "__main__":
    sys.exit(main())
