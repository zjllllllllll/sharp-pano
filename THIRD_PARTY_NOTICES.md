# Third-party source and model inventory

The original license files in each directory remain authoritative. Model files
are downloaded separately and are not included in the Git repository.

| Component | Local source | Recorded upstream/source snapshot |
|---|---|---|
| DepthMaster | `third_party/DepthMaster` | [PolyU-VCLab/DepthMaster](https://github.com/PolyU-VCLab/DepthMaster) `c2ec897fa14803f05f211f6de49571801a4edfa1`; MIT |
| utils3d | `third_party/DepthMaster/third_party/utils3d` | `3fab839f0be9931dac7c8488eb0e1600c236e183`; original LICENSE retained |
| Apple SHARP | `third_party/ml-sharp` | Based on `1eaa046834b81852261262b41b0919f5c1efdd2e`; LICENSE, LICENSE_MODEL and ACKNOWLEDGEMENTS retained |
| LingBot-Depth | `third_party/lingbot-depth` | [Robbyant/lingbot-depth](https://github.com/Robbyant/lingbot-depth), snapshot from the verified repair project; original Apache-2.0 LICENSE and LEGAL.md retained |
| Depth-anchored repair adapters | `repair` | Local Gaussian repair project; original MIT license retained in `repair/LICENSE` |
| GrsAI/Ark client | `providers/studio/image_api.py` | Local repair project snapshot, 2026-09-28; MIT license in `providers/LICENSE` |
| Company GPT/OSS client | `providers/studio/company_image_api.py`, `oss_input.py` | Recovered from repair-project commit `0292541`; MIT license in `providers/LICENSE` |
| Legacy Seedream client | `providers/click_gaussian` | Local `sharp_sd2_test-master` snapshot; its provenance must be retained when distributing |
| Three.js / Spark | `pano-web/package-lock.json` | Pinned npm packages; their installed license files are retained by npm |

Local compatibility changes retained:

- DepthMaster: PyTorch SDPA fallback without flash-attn, lazy evaluation-only
  imports, and local `infer_panoramas.py` used by the application.
- SHARP: `sharp/utils/gsplat.py` accepts either three or four gsplat return values.
- LingBot: list-input fallback without xFormers on Windows.
- Installed Windows runtime: MSVC optimization flags in gsplat and compiler
  output decoding in PyTorch, reproduced by `scripts/patch_runtime.py`.
- Application adapters use checkout-relative paths and explicit provider selection.
- Repair helpers: the optional coverage-percentage gate is disabled by default;
  single-face depth/alpha measurements remain diagnostic. The renderer loads the
  checkout-local SHARP sources and checks that the CUDA backend is available.
- `third_party/SOURCE_SNAPSHOT.json` records the released adapted files, including
  the existing shared-panorama anchoring in the iterative repair stage. Initial
  step 5 separately preserves native SHARP predictions without final reprojection.

SHARP source code and model weights have different licenses. In particular,
`third_party/ml-sharp/LICENSE_MODEL` defines research-only model use; an application
license does not remove those model restrictions. Required model attribution:
“Apple Machine Learning Research Model is licensed under the Apple Machine
Learning Research Model License Agreement.”

The owner's application code is released under the root MIT LICENSE, copyright
2026 zjllllllllll. This does not relicense third-party code or model weights.
No rights to private scene data are granted; historical scene images, private
credentials, model binaries, environments and generated runs are excluded from
this source release.
