"""Verification timeouts must stop compiler children, not leave CPU-bound orphans."""
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from scripts import verify_native_launch as verifier

def test_timeout_stops_descendant_process(tmp_path):
    pidfile=tmp_path/"child.pid"
    child="import os,time; from pathlib import Path; Path("+repr(str(pidfile))+").write_text(str(os.getpid())); time.sleep(30)"
    parent="import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"+repr(child)+"]); time.sleep(30)"
    with pytest.raises(subprocess.TimeoutExpired):
        verifier.run_tool([sys.executable,"-c",parent],timeout=2)
    assert pidfile.exists(), "Child must have started before testing cleanup"
    pid=int(pidfile.read_text())
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        try:
            os.kill(pid,0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(pid,9)
        pytest.fail("Timed-out verification left its child running")

def test_tool_failure_retains_both_output_streams():
    result=verifier.run_tool([sys.executable,"-c",
        "import sys; print('out'); print('error',file=sys.stderr); sys.exit(7)"],timeout=2)
    assert result.returncode==7
    assert result.stdout.strip()=="out"
    assert result.stderr.strip()=="error"
