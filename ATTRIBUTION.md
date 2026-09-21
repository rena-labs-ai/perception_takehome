# Data attribution

The scenes this assignment runs on come from **ARKitScenes**, released by Apple Inc.

> Dehghan, A., Baruch, G., Chen, Z., Feigin, Y., Fu, P., Gebauer, T., Kurz, B., Dimry, T., Joffe, B.,
> Schwartz, A., Sheatsley, R. *ARKitScenes: A Diverse Real-World Dataset for 3D Indoor Scene
> Understanding Using Mobile RGB-D Data.* NeurIPS 2021 Datasets and Benchmarks Track.
> https://github.com/apple/ARKitScenes

`fetch.py` downloads the scenes from Apple's public CDN at run time; they are not redistributed here,
and the ARKitScenes licence in Apple's repository governs your use of them.

Everything under `variant/` is ours and is derived from that data: the camera calibration and
trajectory are Apple's, edited by us and expressed in a coordinate frame of our choosing, and the
object boxes are Apple's 3D object-detection annotations transformed into that same frame. Treat the
contents of `variant/` as a derivative work of ARKitScenes rather than as original data.

The object detections in `variant/detections.json` were produced by running
[OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) (`google/owlv2-base-patch16-ensemble`,
Apache 2.0) over those frames.
