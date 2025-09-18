#!/usr/bin/env python3

import os
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# Root is assumed to be the tudatpy-examples directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "figures_selected")
LOG_DIR = os.path.join(ROOT_DIR, "logs_selected")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Only run scripts from these subfolders
TARGET_DIRS = ["propagation", "estimation", "mission_design"]

def is_example_script(path):
    """Filter out non-example .py files if needed."""
    name = os.path.basename(path)
    if not name.endswith(".py"):
        return False
    if name.startswith("__"):  # skip __init__.py
        return False
    return True

# Wrapper code injected into each subprocess
WRAPPER_CODE = r"""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import runpy

output_dir, script_path = sys.argv[1:3]

def save_all_figs(prefix):
    figs = [plt.figure(i) for i in plt.get_fignums()]
    for idx, fig in enumerate(figs, 1):
        fname = f"{prefix}_fig{idx}.png"
        fig.savefig(os.path.join(output_dir, fname))
    plt.close("all")

# Monkey-patch plt.show()
def _fake_show(*args, **kwargs):
    rel = os.path.relpath(script_path, os.getcwd())
    prefix = rel.replace(os.sep, "_").replace(".py", "")
    save_all_figs(prefix)

plt.show = _fake_show

# Run the script
runpy.run_path(script_path, run_name="__main__")
"""

def run_script(script_path):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Log file path (relative structure preserved)
    rel = os.path.relpath(script_path, ROOT_DIR).replace(os.sep, "_").replace(".py", "")
    log_file = os.path.join(LOG_DIR, f"{rel}.log")

    # Run with working directory set to the script’s folder
    proc = subprocess.run(
        [sys.executable, "-c", WRAPPER_CODE, OUTPUT_DIR, script_path],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(script_path),   # 👈 this is the key line
    )

    # Save stdout + stderr to log file
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(proc.stderr)

    return script_path, proc.returncode, log_file

def find_scripts():
    scripts = []
    for sub in TARGET_DIRS:
        folder = os.path.join(ROOT_DIR, sub)
        for root, _, files in os.walk(folder):
            for fname in sorted(files):
                full = os.path.join(root, fname)
                if is_example_script(full):
                    scripts.append(full)
    return scripts

def main(num_workers=None):
    scripts = find_scripts()
    print(f"Found {len(scripts)} scripts in {', '.join(TARGET_DIRS)}")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for s in scripts:
            rel = os.path.relpath(s, ROOT_DIR)
            print(f"[RUN]  {rel}")
            futures[executor.submit(run_script, s)] = s

        for future in as_completed(futures):
            script_path, returncode, log_file = future.result()
            rel = os.path.relpath(script_path, ROOT_DIR)
            if returncode == 0:
                print(f"[OK]   {rel} (log: {os.path.basename(log_file)})")
            else:
                print(f"[FAIL] {rel} (see {os.path.basename(log_file)})")

    print(f"\nAll done.\nFigures saved in: {OUTPUT_DIR}\nLogs saved in: {LOG_DIR}")

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(num_workers=workers)
