"""Regression tests for the 2026-06-10 review critical-bug batch (docs/reviews/2026-06-10/02-bugs.md)."""
import os
import subprocess
import sys

import pytest
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_runner_help_works_without_distutils():
    # Finding P5: `from distutils.util import strtobool` crashes on py3.12+ (distutils removed).
    # '--help' exercises the import-time crash point (the old line-1 distutils import).
    res = subprocess.run([sys.executable, os.path.join(REPO, 'runner.py'), '--help'],
                         capture_output=True, text=True, timeout=120, cwd=REPO)
    assert res.returncode == 0, res.stderr
    assert 'distutils' not in res.stderr
    assert 'usage' in res.stdout.lower()


def test_strtobool_semantics():
    # Direct unit test: runner.py is import-safe (all side effects live under __main__).
    sys.path.insert(0, REPO)
    try:
        from runner import strtobool
    finally:
        sys.path.remove(REPO)
    for token in ('y', 'yes', 't', 'true', 'on', '1', 'YES', ' True '):
        assert strtobool(token) == 1
    for token in ('n', 'no', 'f', 'false', 'off', '0', 'OFF'):
        assert strtobool(token) == 0
    with pytest.raises(ValueError):
        strtobool('maybe')
