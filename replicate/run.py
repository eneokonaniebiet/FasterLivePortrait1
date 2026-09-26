import os
import shutil
import subprocess
import tempfile
from pathlib import Path as SysPath

from cog import BasePredictor, Input, Path

ROOT = SysPath(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"
PLUGIN = CHECKPOINTS / "liveportrait_onnx" / "libgrid_sample_3d_plugin.so"


class Runner(BasePredictor):
    """Replicate adapter for the existing FasterLivePortrait1 TensorRT pipeline."""

    def setup(self):
        self._prepare_runtime()
        import torch
        from omegaconf import OmegaConf
        from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("FasterLivePortrait Replicate runner requires CUDA")

        cfg = OmegaConf.load(str(ROOT / "configs" / "trt_infer.yaml"))
        cfg.infer_params.flag_pasteback = True
        cfg.infer_params.flag_do_crop = True
        cfg.infer_params.flag_stitching = True
        cfg.infer_params.flag_relative_motion = True
        self.pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)

    def _prepare_runtime(self):
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        human = CHECKPOINTS / "liveportrait_onnx"
        if not human.exists() or not any(human.glob("*.onnx")):
            subprocess.run([
                "huggingface-cli", "download", "warmshao/FasterLivePortrait",
                "--local-dir", str(CHECKPOINTS)
            ], check=True)

        human.mkdir(parents=True, exist_ok=True)
        if not PLUGIN.exists():
            plugin_root = SysPath("/tmp/grid-sample3d-trt-plugin")
            if not plugin_root.exists():
                subprocess.run([
                    "git", "clone", "--depth", "1",
                    "https://github.com/SeanWangJS/grid-sample3d-trt-plugin",
                    str(plugin_root)
                ], check=True)
            cmake_file = plugin_root / "CMakeLists.txt"
            text = cmake_file.read_text()
            text = text.replace(
                'CUDA_ARCHITECTURES "70;80;86;89"',
                'CUDA_ARCHITECTURES "75;80;86"'
            )
            cmake_file.write_text(text)
            build = plugin_root / "build"
            build.mkdir(exist_ok=True)
            subprocess.run([
                "cmake", "..", "-DTensorRT_ROOT=/usr/local/tensorrt"
            ], cwd=build, check=True)
            subprocess.run(["make", "-j2"], cwd=build, check=True)
            shutil.copy2(build / "libgrid_sample_3d_plugin.so", PLUGIN)

        required = [
            human / "warping_spade-fix.trt",
            human / "motion_extractor.trt",
            human / "landmark.trt",
            human / "retinaface_det_static.trt",
            human / "face_2dpose_106_static.trt",
            human / "appearance_feature_extractor.trt",
            human / "stitching.trt",
            human / "stitching_eye.trt",
            human / "stitching_lip.trt",
        ]
        if not all(p.exists() for p in required):
            subprocess.run(
                ["bash", str(ROOT / "scripts" / "all_onnx2trt.sh")],
                cwd=ROOT,
                check=True,
            )

    def predict(
        self,
        source_image: Path = Input(description="Source portrait image."),
        driving_video: Path = Input(description="Driving video containing the face motion."),
        fps: int = Input(description="Output FPS.", default=30, ge=1, le=60),
        driving_multiplier: float = Input(description="Motion multiplier.", default=1.0, ge=0.1, le=3.0),
    ) -> Path:
        import cv2

        work = SysPath(tempfile.mkdtemp(prefix="flp-"))
        src = work / "source.png"
        dri = work / "driving.mp4"
        shutil.copy2(str(source_image), src)
        shutil.copy2(str(driving_video), dri)

        self.pipe.init_vars()
        self.pipe.update_cfg({
            "driving_multiplier": driving_multiplier,
            "flag_relative_motion": True,
            "flag_stitching": True,
            "flag_pasteback": True,
        })

        if not self.pipe.prepare_source(str(src), realtime=False):
            raise RuntimeError("No face detected in source image")

        cap = cv2.VideoCapture(str(dri))
        if not cap.isOpened():
            raise RuntimeError("Unable to open driving video")

        width = self.pipe.src_imgs[0].shape[1]
        height = self.pipe.src_imgs[0].shape[0]
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        out_fps = fps or int(source_fps or 30)

        out = work / "output.mp4"
        writer = cv2.VideoWriter(
            str(out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            out_fps,
            (width, height),
        )

        frame_index = 0
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, _, out_org, _ = self.pipe.run(
                frame,
                self.pipe.src_imgs[0],
                self.pipe.src_infos[0],
                first_frame=(frame_index == 0),
            )
            frame_index += 1
            if out_org is None:
                continue
            writer.write(cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR))
            written += 1

        cap.release()
        writer.release()

        if written == 0:
            raise RuntimeError("No frames were rendered")

        return Path(str(out))
