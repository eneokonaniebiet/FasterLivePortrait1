"""Cog entrypoint for FasterLivePortrait file-based animation.

This runner deliberately uses the repository's supported ONNX inference config.
It does not claim to provide continuous live-camera streaming.
"""
from pathlib import Path as LocalPath
import os
import subprocess
import tempfile

from cog import BaseRunner, Input, Path
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf


ROOT = LocalPath(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
CONFIG = ROOT / "configs" / "onnx_infer.yaml"
RESULTS_DIR = ROOT / "results"
WEIGHTS_REPO = os.getenv("FLIP_HF_REPO", "warmshao/FasterLivePortrait")


def log(message: str) -> None:
    """Write stage markers directly to the Cog worker log."""
    print(f"[FasterLivePortrait runner] {message}", flush=True)


class Runner(BaseRunner):
    def setup(self) -> None:
        log("Setup started.")
        if not CONFIG.exists():
            raise RuntimeError(f"Missing inference config: {CONFIG}")
        required = CHECKPOINT_DIR / "liveportrait_onnx" / "motion_extractor.onnx"
        if not required.exists():
            log(f"Downloading ONNX weights from {WEIGHTS_REPO}.")
            snapshot_download(
                repo_id=WEIGHTS_REPO,
                local_dir=str(CHECKPOINT_DIR),
                allow_patterns=["liveportrait_onnx/*"],
            )
        if not required.exists():
            raise RuntimeError(
                "Checkpoint download completed but expected ONNX weights are missing: "
                f"{required}. Check the Hugging Face repository layout."
            )
        log("Setup checks passed; required motion_extractor.onnx is present.")

    def run(
        self,
        source_image: Path = Input(description="Portrait image to animate"),
        driving_video: Path = Input(description="Video whose facial motion drives the portrait"),
        fps: int = Input(description="Output FPS (reserved for future output normalization)", default=30, ge=1, le=60),
        driving_multiplier: float = Input(description="Motion strength (reserved for future pipeline support)", default=1.0, ge=0.1, le=3.0),
    ) -> Path:
        log("Prediction started.")
        if not LocalPath(str(source_image)).is_file():
            raise RuntimeError("The supplied source image is not readable.")
        if not LocalPath(str(driving_video)).is_file():
            raise RuntimeError("The supplied driving video is not readable.")
        log("Input image and driving video are readable.")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        before = set(RESULTS_DIR.rglob("*-org.mp4"))
        inference_config = OmegaConf.load(str(CONFIG))
        inference_config.infer_params.driving_multiplier = float(driving_multiplier)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_file:
            temp_config_path = config_file.name
            OmegaConf.save(inference_config, temp_config_path)

        command = [
            "python",
            str(ROOT / "run.py"),
            "--src_image", str(source_image),
            "--dri_video", str(driving_video),
            "--cfg", temp_config_path,
            "--paste_back",
        ]
        log("Starting upstream inference process; its stdout/stderr will stream to worker logs.")
        try:
            try:
                # Inherit the worker's stdout/stderr instead of buffering all
                # output in memory. This makes progress and tracebacks visible
                # in prediction logs while inference is running.
                completed = subprocess.run(
                    command,
                    cwd=str(ROOT),
                    text=True,
                    stdout=None,
                    stderr=subprocess.STDOUT,
                    timeout=60 * 60,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "FasterLivePortrait inference exceeded the 60-minute runner timeout. "
                    "Check the streamed prediction logs for the last completed stage."
                ) from exc
        finally:
            try:
                os.unlink(temp_config_path)
            except OSError:
                pass

        if completed.returncode != 0:
            raise RuntimeError(
                f"FasterLivePortrait inference process exited with code {completed.returncode}. "
                "See the streamed prediction logs above for the underlying error."
            )
        log("Inference process exited successfully; checking for its output video.")

        candidates = [p for p in RESULTS_DIR.rglob("*-org.mp4") if p not in before]
        if not candidates:
            # If the process reused a timestamped result path, select the newest
            # output file, but never return an arbitrary non-video artifact.
            candidates = list(RESULTS_DIR.rglob("*-org.mp4"))
        if not candidates:
            raise RuntimeError(
                "Inference exited successfully but no *-org.mp4 output was found under "
                f"{RESULTS_DIR}. Review the streamed inference logs."
            )
        output = max(candidates, key=lambda p: p.stat().st_mtime)
        if output.stat().st_size == 0:
            raise RuntimeError(f"Inference produced an empty output video: {output}")
        log(f"Output video found ({output.stat().st_size} bytes); starting FPS normalization.")

        normalized_output = output.with_name(output.stem + f"-{int(fps)}fps.mp4")
        try:
            normalize = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(output),
                    "-vf", f"fps={int(fps)}",
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(normalized_output),
                ],
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60 * 20,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Output FPS normalization exceeded the 20-minute timeout.") from exc

        if normalize.returncode != 0 or not normalized_output.exists() or normalized_output.stat().st_size == 0:
            details = (normalize.stdout or "")[-8000:]
            raise RuntimeError(
                "Output FPS normalization failed. FFmpeg output follows:\n" + details
            )
        log(f"Prediction finished successfully. Returning {normalized_output.name}.")
        return Path(str(normalized_output))
