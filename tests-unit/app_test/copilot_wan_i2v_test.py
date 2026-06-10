import subprocess
import sys
from pathlib import Path


def test_wan_i2v_prompt_check_script():
    script = Path(__file__).with_name("run_copilot_wan_i2v_check.py")
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
