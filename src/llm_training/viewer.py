"""Shared training dashboard and visible launch attachment."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
def render_progress(status: dict) -> str:
    """Render the complete terminal dashboard without fixed token assumptions."""
    step = status.get("step", 0)
    total = status.get("total_steps", 1)
    percentage = min(100.0, 100.0 * step / max(total, 1))
    filled = int(percentage / 2)
    eta = max(0, int(status.get("eta_seconds", 0)))
    eta_text = f"{eta // 3600:02d}:{eta // 60 % 60:02d}:{eta % 60:02d}"
    train_loss = status.get("training_loss")
    validation_loss = status.get("validation_loss")
    loss = "--" if train_loss is None else f"{train_loss:.5f}"
    validation = "--" if validation_loss is None else f"{validation_loss:.5f}"
    return "\n".join([
        "  SPECIALIST  |  FROZEN EMBEDDING CLASSIFICATION",
        "  " + "-" * 76,
        "",
        f"  {'#' * filled}{'-' * (50 - filled)}  {percentage:5.1f}%  ETA {eta_text}",
        "",
        f"  STATUS      {status.get('state', 'STARTING')}",
        f"  HEAD        {status.get('kind', '--')}",
        f"  STEPS       {step} / {total}",
        f"  SPEED       {status.get('tokens_per_second', 0):.0f} tok/s",
        f"  TRAIN LOSS  {loss}",
        f"  VAL LOSS    {validation}",
        f"  TOKENS      {status.get('total_tokens_seen', 0):,}",
        "",
        "  CHECKPOINT",
        f"  {status.get('checkpoint_path', '--')}",
        "",
        "  FP32 head | tokens represent cached requests presented to the head",
    ])


@contextmanager
def attached_viewer(*, metrics_path: Path, checkpoint_path: Path, total_steps: int, title: str):
    """Attach the required Kitty viewer before an operator training launch."""
    if not shutil.which("kitty") or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("A visible Kitty viewer is required before training.")
    with tempfile.TemporaryDirectory(prefix="llm-training-viewer-") as directory:
        marker = Path(directory) / "attached"
        process = subprocess.Popen([
            "kitty", "--override", "remember_window_size=no",
            "--override", "initial_window_width=100c", "--override", "initial_window_height=22c",
            "--title", title, sys.executable, "-m", "llm_training.viewer",
            "--metrics", str(metrics_path.resolve()), "--checkpoint", str(checkpoint_path.resolve()),
            "--steps", str(total_steps), "--marker", str(marker), "--owner", str(os.getpid()),
        ])
        try:
            deadline = time.monotonic() + 15
            while not marker.is_file():
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Kitty failed to attach the required training viewer.")
                time.sleep(0.1)
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main():
    """Watch one run's measured metrics; never launch or import a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--owner", type=int, required=True)
    args = parser.parse_args()
    args.marker.write_text(str(os.getpid()))
    while True:
        value = {}
        if args.metrics.is_file():
            # Read only the tail, even when training metrics have grown large.
            with args.metrics.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 65536))
                for line in reversed(stream.read().splitlines()):
                    try:
                        value = json.loads(line)
                        break
                    except (ValueError, UnicodeError):
                        continue
        step = value.get("optimizer_step", 0)
        value.update(state="RUNNING" if step else "STARTING", kind="decoder", step=step,
                     total_steps=args.steps, checkpoint_path=str(args.checkpoint),
                     eta_seconds=value.get("step_time_seconds", 0) * max(0, args.steps - step))
        text = render_progress(value).replace("SPECIALIST  |  FROZEN EMBEDDING CLASSIFICATION", "LLM  |  NATIVE MODEL TRAINING")
        text = text.replace("FP32 head | tokens represent cached requests presented to the head", "Speed and tokens are measured by the trainer")
        print("\033[2J\033[H" + text, flush=True)
        try:
            os.kill(args.owner, 0)
        except ProcessLookupError:
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
