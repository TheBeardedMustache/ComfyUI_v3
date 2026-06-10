import subprocess
import sys
from pathlib import Path


def test_wan_i2v_prompt_check_script():
    script = Path(__file__).with_name("run_copilot_wan_i2v_check.py")
    recipe_script = Path(__file__).with_name("run_copilot_wan_recipe_validate.py")
    repo_root = Path(__file__).resolve().parents[2]
    for script_path in (script, recipe_script):
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
