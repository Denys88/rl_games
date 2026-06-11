"""Regression tests for the 2026-06-10 review critical-bug batch (docs/reviews/2026-06-10/02-bugs.md)."""
import copy
import os
import subprocess
import sys

import pytest
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_runner_help_works_without_distutils():
    # Finding P5: `from distutils.util import strtobool` crashes on py3.12+ (distutils removed).
    # `--help` exercises the full argparse setup including the strtobool lambda default path.
    res = subprocess.run([sys.executable, os.path.join(REPO, 'runner.py'), '--help'],
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert 'distutils' not in res.stderr
