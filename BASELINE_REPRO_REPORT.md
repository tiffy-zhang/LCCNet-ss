# LCCNet Baseline Reproduction Report

Date: 2026-03-26
Project: LCCNet baseline reproduction (KITTI odometry)
Goal: Reproduce original supervised baseline pipeline without redesigning model/loss/data flow.

## 1) Reproduction Scope

Completed:
- Baseline codebase understanding and key file mapping
- Environment compatibility setup (minimal changes)
- Evaluation pipeline run-through (single and multi-range iterative)
- Training smoke test run-through on GPU0

Not included in this stage:
- Self-supervised redesign
- Architecture changes
- New loss functions
- Pipeline optimization

## 2) Key Code Entry Points

- Training entry: train_with_sacred.py
- Evaluation entry: evaluate_calib.py
- Dataset loader: DatasetLidarCamera.py
- Model definition: models/LCCNet.py
- Loss definitions: losses.py
- Correlation CUDA extension: models/correlation_package/

## 3) Verified Runtime Environment

- Python: 3.12.4
- PyTorch: 2.3.1+cu121
- Torchvision: 0.18.1+cu121
- Sacred: 0.8.7
- OpenCV: 4.8.1
- Open3D: 0.19.0
- TensorboardX: 2.6.4
- skimage: 0.23.2

System GPU/CUDA:
- GPU: NVIDIA GeForce RTX 4090 (8 cards available)
- NVIDIA Driver: 550.120
- System CUDA: 12.4

## 4) Dataset Verification (KITTI Odometry)

Data root used:
- /home/zmy/datasets/kitti/dataset

Verified structure per sequence:
- sequences/XX/image_2/*.png
- sequences/XX/velodyne/*.bin
- sequences/XX/calib.txt

Checked sequence counts (00-10): image and velodyne counts matched for all tested sequences.

## 5) Pretrained Weights

Used weight files:
- /home/zmy/LCCNet-new/pretrained/kitti_iter1.tar
- /home/zmy/LCCNet-new/pretrained/kitti_iter2.tar
- /home/zmy/LCCNet-new/pretrained/kitti_iter3.tar
- /home/zmy/LCCNet-new/pretrained/kitti_iter4.tar
- /home/zmy/LCCNet-new/pretrained/kitti_iter5.tar

In project root, pretrained is linked as:
- pretrained -> ../pretrained

## 6) Minimal Compatibility Fixes Applied

These are runtime compatibility patches only; baseline algorithm behavior remains unchanged.

1. Sacred read-only config compatibility
- Avoid mutating _config in train/eval entry code; use local variables instead.

2. GPU0 and single-GPU compatibility
- Ensure CUDA_VISIBLE_DEVICES default is set before torch import.
- Use DataParallel only when cuda device count > 1.

3. Correlation extension compatibility (PyTorch 2.x + Ada GPU)
- Update custom autograd wrapper to static Function API.
- Update extension compile flags to C++17.
- Add sm_89 compile target.
- Keep extension import package-relative fallback.
- Update pyproject build requirements to modern torch for repeatable build in this environment.

4. mathutils compatibility layer
- Add local mathutils.py shim for APIs used by baseline code path.
- Add Matrix length protocol and CUDA tensor to numpy conversion compatibility.

5. Dataset stale import fix
- Remove non-existent read_calib_file import from DatasetLidarCamera.py.

## 7) Evaluation Reproduction Results

### A) Single iteration run (iterative_method=single)
Command pattern:
- python evaluate_calib.py with data_folder=/home/zmy/datasets/kitti/dataset test_sequence=0 iterative_method=single

Status:
- Completed successfully
- Runtime about 5m35s

Key metrics:
- Iteration 1 mean translation error: 15.7232 cm
- Iteration 1 mean rotation error: 1.2469 deg

### B) Full iterative run (iterative_method=multi_range)
Command pattern:
- python evaluate_calib.py with data_folder=/home/zmy/datasets/kitti/dataset test_sequence=0 iterative_method=multi_range

Status:
- Completed successfully
- Runtime about 14m54s

Key final metrics:
- Iteration 5 mean translation error: 1.3186 cm
- Iteration 5 mean rotation error: 0.2246 deg

Output artifact:
- /home/zmy/LCCNet-new/output/multi_range/results.txt

## 8) Training Smoke Test (GPU0)

Command pattern:
- python train_with_sacred.py with data_folder=/home/zmy/datasets/kitti/dataset val_sequence=0 epochs=1 starting_epoch=1 batch_size=8 num_worker=2 resume=False weights=./pretrained/kitti_iter5.tar

Status:
- Training loop entered successfully
- Dataloader + forward + loss + backward path verified by stable iteration logs
- Observed logs reached at least Iter 50 / 100 / 150

Note:
- Smoke run was intentionally stopped after confirming stable training iterations to avoid long GPU occupation.

## 9) Files Safe to Keep Unchanged (for baseline fidelity)

- models/LCCNet.py (network architecture)
- losses.py (baseline loss definitions)
- core dataset semantics in DatasetLidarCamera.py (sampling/labels/projection flow)
- quaternion_distances.py and core geometry utilities in utils.py (except compatibility-only interactions)

## 10) Files Likely to Modify Later (self-supervised stage)

- train_with_sacred.py (training supervision signals and objective wiring)
- evaluate_calib.py (optional iterative policy packaging and diagnostics)
- DatasetLidarCamera.py (if adding unlabeled/self-supervised data fields)
- New self-supervised losses/modules (future stage only; not part of this baseline report)

## 11) Conclusion

Baseline reproduction target is achieved for the current stage:
- Environment configured and validated
- Evaluation pipeline fully run-through
- Training smoke path verified on GPU0

Ready for the next stage: controlled self-supervised extension on top of this validated baseline.
