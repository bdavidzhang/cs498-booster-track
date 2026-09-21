"""Run course training on Modal, mirroring scripts/train.sbatch on DeltaAI.

DeltaAI's ghx4 job (see train.sbatch): 1x GH200 (Hopper GPU), 16 CPUs,
~111 GiB RAM (DefMemPerGPU=113560M), 4 h wall time. Modal has no GH200, so we
use an H100 (the same Hopper GPU generation) with the same CPU/RAM/time.

One-time setup:
  uvx modal setup
  uvx modal secret create wandb WANDB_API_KEY=<key>   # or run with COURSE_NO_WANDB=1

Train (from course/; --detach keeps it running if your terminal drops):
  uvx modal run --detach modal_train.py::main --task Course-Cartpole-Double-Balance
  uvx modal run --detach modal_train.py::main --task Course-Cartpole-Double-Swingup

Pull logs into this repo's logs/ (safe to rerun; only fetches new/changed files):
  uvx modal run modal_train.py::download                      # everything
  uvx modal run modal_train.py::download --prefix rsl_rl/cartpole_double/<run>

Layout mirrors the cluster:
  logs/modal-<task>-<timestamp>.out     stdout/stderr (like slurm-<jobid>.out)
  logs/rsl_rl/<experiment>/<run>/...    checkpoints, params/, tfevents
"""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import modal

HERE = Path(__file__).parent
REMOTE_REPO = "/root/course"
VOL_MOUNT = "/vol"
USE_WANDB = os.environ.get("COURSE_NO_WANDB") != "1"

app = modal.App("course-train")
logs_vol = modal.Volume.from_name("course-train-logs", create_if_missing=True)

image = (
  modal.Image.debian_slim(python_version="3.13")
  # git: mjlab is a git dependency pinned in pyproject.toml.
  # libegl1/libgl1: mjlab sets MUJOCO_GL=egl, and `import mujoco` then loads
  # libEGL at import time; debian_slim doesn't ship it.
  .apt_install("git", "libegl1", "libgl1")
  .pip_install("uv")
  .env({"UV_PYTHON": "python3.13", "UV_FROZEN": "1", "UV_LINK_MODE": "copy"})
  .workdir(REMOTE_REPO)
  # Dependencies first so code edits don't reinstall torch/mjlab.
  .add_local_file(HERE / "pyproject.toml", f"{REMOTE_REPO}/pyproject.toml", copy=True)
  .add_local_file(HERE / "uv.lock", f"{REMOTE_REPO}/uv.lock", copy=True)
  .add_local_file(HERE / "README.md", f"{REMOTE_REPO}/README.md", copy=True)
  .run_commands("uv sync --frozen --no-install-project")
  # Then the course package itself; installing the project registers the
  # mjlab.tasks entry point that makes Course-* task ids visible.
  .add_local_dir(
    HERE / "src", f"{REMOTE_REPO}/src", copy=True, ignore=["**/__pycache__", "**/*.pyc"]
  )
  .run_commands("uv sync --frozen")
)


def _commit_periodically(stop: threading.Event, every_s: float = 60.0) -> None:
  # Persist checkpoints while training runs, so download works mid-run.
  while not stop.wait(every_s):
    try:
      logs_vol.commit()
    except Exception as e:  # a failed commit shouldn't kill training
      print(f"[modal_train] volume commit failed: {e}", flush=True)


@app.function(
  image=image,
  gpu="H100",
  cpu=16,
  memory=113560,
  timeout=4 * 60 * 60,
  volumes={VOL_MOUNT: logs_vol},
  secrets=[modal.Secret.from_name("wandb")] if USE_WANDB else [],
)
def train(task: str, extra_args: list[str]) -> None:
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  out_path = Path(VOL_MOUNT) / f"modal-{task}-{stamp}.out"
  cmd = ["uv", "run", "train", task, "--log-root", f"{VOL_MOUNT}/rsl_rl", *extra_args]
  if not USE_WANDB:
    cmd += ["--agent.logger", "tensorboard"]

  env = dict(os.environ, WANDB_DIR=VOL_MOUNT, PYTHONUNBUFFERED="1")
  if "WANDB_API_KEY" in env:
    # wandb rejects keys with surrounding whitespace (easy to paste into a secret).
    env["WANDB_API_KEY"] = env["WANDB_API_KEY"].strip()
  stop = threading.Event()
  threading.Thread(target=_commit_periodically, args=(stop,), daemon=True).start()

  with out_path.open("w") as out:

    def log(line: str) -> None:
      print(line, end="", flush=True)
      out.write(line)
      out.flush()

    log(f"== {datetime.now()} modal job on {os.uname().nodename} ==\n")
    log(f"== task: {task}  extra args: {shlex.join(extra_args)}\n")
    log(f"== cmd: {shlex.join(cmd)}\n")
    log(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

    proc = subprocess.Popen(
      cmd, cwd=REMOTE_REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.stdout is not None
    for line in proc.stdout:
      log(line)
    code = proc.wait()
    log(f"== {datetime.now()} done (exit {code}) ==\n")

  stop.set()
  logs_vol.commit()
  if code != 0:
    raise RuntimeError(f"train exited with code {code}; see {out_path.name}")


@app.local_entrypoint()
def main(task: str, num_envs: int = 4096, extra: str = "") -> None:
  """Launch one training run. Extra train flags go in --extra "..."."""
  extra_args = ["--env.scene.num-envs", str(num_envs), *shlex.split(extra)]
  train.remote(task, extra_args)


@app.local_entrypoint()
def download(prefix: str = "", dest: str = str(HERE / "logs")) -> None:
  """Mirror the Modal log volume into course/logs/, skipping unchanged files.

  --prefix limits it to paths starting with that string, e.g.
  rsl_rl/cartpole_double/2026-09-16_21-03-11 for a single run.
  """
  dest_root = Path(dest)
  prefix = prefix.strip("/")
  fetched = skipped = 0
  for entry in logs_vol.listdir("/", recursive=True):
    if entry.type != modal.volume.FileEntryType.FILE:
      continue
    if not entry.path.lstrip("/").startswith(prefix):
      continue
    local = dest_root / entry.path.lstrip("/")
    if local.exists() and local.stat().st_size == entry.size:
      skipped += 1
      continue
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(local.name + ".part")
    with tmp.open("wb") as f:
      for chunk in logs_vol.read_file(entry.path):
        f.write(chunk)
    tmp.replace(local)
    fetched += 1
  print(f"downloaded {fetched} file(s), {skipped} unchanged -> {dest_root}")
