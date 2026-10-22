"""Cog entrypoint for FasterLivePortrait file-based animation.

This runner deliberately uses the repository's supported ONNX inference config.
It does not claim to provide continuous live-camera streaming.
"""
from pathlib import Path as LocalPath
import os
import subprocess

import cv2
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
        source_path = LocalPath(str(source_image))
        driving_path = LocalPath(str(driving_video))
        if not source_path.is_file() or source_path.stat().st_size == 0:
            raise RuntimeError("The supplied source image is missing, empty, or unreadable.")
        if not driving_path.is_file() or driving_path.stat().st_size == 0:
            raise RuntimeError("The supplied driving video is missing, empty, or unreadable.")

        # Fail early on malformed uploads rather than spending GPU time in inference.
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise RuntimeError(
                "The source_image could not be decoded as an image. Upload a valid JPG or PNG portrait."
            )
        image_height, image_width = image.shape[:2]
        if image_width < 64 or image_height < 64:
            raise RuntimeError(
                f"The source image is too small ({image_width}x{image_height}); use an image at least 64x64."
            )
        del image
        log(f"Source image validated ({image_width}x{image_height}).")

        capture = cv2.VideoCapture(str(driving_path))
        try:
            if not capture.isOpened():
                raise RuntimeError(
                    "The driving_video could not be opened. Upload a supported video such as MP4 (H.264)."
                )
            video_fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            ok, first_frame = capture.read()
            if not ok or first_frame is None or first_frame.size == 0:
                raise RuntimeError(
                    "The driving_video opened but no readable frame was found. Try a different video file."
                )
            frame_height, frame_width = first_frame.shape[:2]
            if frame_width < 64 or frame_height < 64:
                raise RuntimeError(
                    f"The driving video frame is too small ({frame_width}x{frame_height}); use video at least 64x64."
                )
            if video_fps <= 0:
                raise RuntimeError(
                    "The driving video has no usable frame-rate metadata. Re-export it as a standard MP4 video."
                )
            if frame_count > 0 and frame_count < 2:
                raise RuntimeError("The driving video must contain at least two frames.")
        finally:
            capture.release()
        log(
            f"Driving video validated ({frame_width}x{frame_height}, "
            f"{video_fps:.2f} FPS, reported frames={int(frame_count) if frame_count > 0 else 'unknown'})."
        )

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

        # Confirm the normalized file is a decodable video before returning it.
        result_capture = cv2.VideoCapture(str(normalized_output))
        try:
            if not result_capture.isOpened():
                raise RuntimeError("FPS normalization produced a file that OpenCV cannot open.")
            result_ok, result_frame = result_capture.read()
            if not result_ok or result_frame is None or result_frame.size == 0:
                raise RuntimeError("FPS normalization produced a video with no readable frames.")
            result_width = int(result_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            result_height = int(result_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            result_frame_count = int(result_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            result_capture.release()
        log(
            f"Output validated ({result_width}x{result_height}, "
            f"reported frames={result_frame_count}, bytes={normalized_output.stat().st_size})."
        )
        log(f"Prediction finished successfully. Returning {normalized_output.name}.")
        return Path(str(normalized_output))
