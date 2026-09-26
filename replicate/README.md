# Replicate test adapter

This branch packages the existing FasterLivePortrait1 TensorRT pipeline for a first Replicate test.

Input:
- source portrait image
- driving video

Output:
- rendered MP4

The adapter is intentionally separate from the existing Render/Kémzy path.

Important:
- The current FasterLivePortrait implementation requires TensorRT 8.x and the custom 3D grid-sample TensorRT plugin.
- This first adapter prepares those components at runtime and downloads the upstream FasterLivePortrait checkpoints from Hugging Face.
- For production, the checkpoints and TensorRT engines should be baked into the Replicate model image or managed weights so they are not rebuilt/downloaded on every cold start.

Replicate model target:
r8.im/eneokonaniebiet/fasterliveportrait1
