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


class Runner(BaseRunner):
    def setup(self) -> None:
        if not CONFIG.exists():
            raise RuntimeError(f"Missing inference config: {CONFIG}")
        required = CHECKPOINT_DIR / "liveportrait_onnx" / "motion_extractor.onnx"
        if not required.exists():
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

    def run(
        self,
        source_image: Path = Input(description="Portrait image to animate"),
        driving_video: Path = Input(description="Video whose facial motion drives the portrait"),
        fps: int = Input(description="Output FPS (reserved for future output normalization)", default=30, ge=1, le=60),
        driving_multiplier: float = Input(description="Motion strength (reserved for future pipeline support)", default=1.0, ge=0.1, le=3.0),
    ) -> Path:
        # Use a temporary copy of the upstream config so motion strength is
        # applied without mutating the repository's default config.
        if not LocalPath(str(source_image)).is_file():
            raise RuntimeError("The supplied source image is not readable.")
        if not LocalPath(str(driving_video)).is_file():
            raise RuntimeError("The supplied driving video is not readable.")

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
        try:
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60 * 60,
            )
        finally:
            try:
                os.unlink(temp_config_path)
            except OSError:
                pass
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

        normalized_output = output.with_name(output.stem + f"-{int(fps)}fps.mp4")
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
        )
        if normalize.returncode != 0 or not normalized_output.exists() or normalized_output.stat().st_size == 0:
            raise RuntimeError("Output FPS normalization failed:\\n" + normalize.stdout[-8000:])
        return Path(str(normalized_output))
