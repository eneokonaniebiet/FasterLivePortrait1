"""Cog entrypoint for FasterLivePortrait file-based animation.

This runner deliberately uses the repository's supported ONNX inference config.
It does not claim to provide continuous live-camera streaming.
"""
from pathlib import Path as LocalPath
import os
import subprocess
import time
from typing import Optional

from cog import BaseRunner, Input, Path
from huggingface_hub import snapshot_download


ROOT = LocalPath(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
CONFIG = ROOT / "configs" / "onnx_infer.yaml"
RESULTS_DIR = ROOT / "results"
WEIGHTS_REPO = os.getenv("FLIP_HF_REPO", "warmshao/FasterLivePortrait")


class Runner(BaseRunner):
    def setup(self) -> None:
        if not CONFIG.exists():
            raise RuntimeError(f"Missing inference config: {CONFIG}")
        required = CHECKPOINT_DIR / "liveportrait_onnx" / "motion_extractor.onnx"
        if not required.exists():
            snapshot_download(
                repo_id=WEIGHTS_REPO,
                local_dir=str(CHECKPOINT_DIR),
                local_dir_use_symlinks=False,
            )
        if not required.exists():
            raise RuntimeError(
                "Checkpoint download completed but expected ONNX weights are missing: "
                f"{required}. Check the Hugging Face repository layout."
            )

    def run(
        self,
        source_image: Path = Input(description="Portrait image to animate"),
        driving_video: Path = Input(description="Video whose facial motion drives the portrait"),
        fps: int = Input(description="Output FPS (reserved for future output normalization)", default=30, ge=1, le=60),
        driving_multiplier: float = Input(description="Motion strength (reserved for future pipeline support)", default=1.0, ge=0.1, le=3.0),
    ) -> Path:
        # The upstream command currently derives FPS from the driving video and
        # does not expose driving_multiplier. Keep the input schema compatible,
        # but do not silently claim these options are applied.
        if not LocalPath(str(source_image)).is_file():
            raise RuntimeError("The supplied source image is not readable.")
        if not LocalPath(str(driving_video)).is_file():
            raise RuntimeError("The supplied driving video is not readable.")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        before = set(RESULTS_DIR.rglob("*-org.mp4"))
        command = [
            "python",
            str(ROOT / "run.py"),
            "--src_image", str(source_image),
            "--dri_video", str(driving_video),
            "--cfg", str(CONFIG),
            "--paste_back",
        ]
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60 * 60,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "FasterLivePortrait inference failed. Last output:\n"
                + completed.stdout[-12000:]
            )

        candidates = [p for p in RESULTS_DIR.rglob("*-org.mp4") if p not in before]
        if not candidates:
            # If the process reused a timestamped result path, select the newest
            # output file, but never return an arbitrary non-video artifact.
            candidates = list(RESULTS_DIR.rglob("*-org.mp4"))
        if not candidates:
            raise RuntimeError(
                "Inference completed without an output video. Logs:\n"
                + completed.stdout[-12000:]
            )
        output = max(candidates, key=lambda p: p.stat().st_mtime)
        if output.stat().st_size == 0:
            raise RuntimeError("Inference produced an empty output video.")
        return Path(str(output))
