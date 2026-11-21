import os

import pytest


@pytest.fixture(autouse=True)
def no_real_execv(monkeypatch):
    """A PBT restart re-execs in the step that decides it: a test that gets there without
    capturing os.execv (tests/test_pbt_records.py: `execs`) would replace pytest with the restart
    command. Tests that patch os.execv themselves override this."""
    def execv(path, argv):
        raise RuntimeError(f"a test reached os.execv({path!r}, ...) without capturing it")

    monkeypatch.setattr(os, "execv", execv)
