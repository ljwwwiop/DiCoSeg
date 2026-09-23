# 🚧 [Diffusion-Based Contextual Reconstruction for Point Cloud Segmentation with Limited Annotations](https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=svW3qjkAAAAJ&citation_for_view=svW3qjkAAAAJ:zYLM7Y9cAGgC)

🎯 **DiCoSeg Author** : Jiawei Lian, Zhengxue Wang, Wentao Qu, Haobo Jiang, Le Hui, Jian Yang



🔬 Tackling 3D segmentation under annotation scarcity? DiCoSeg leverages diffusion models for **context-aware reconstruction** with just a few labels.  
⚡️ Single-step inference, strong generalization, and robust results across benchmarks.

📈 Currently under submission — stay tuned for updates!  
📬 *If you're reviewing this... be kind* 😄

📎 Initial code is now available. See the update notes below.

| Settings | Method               | mIoU  | Where   |
|----------|----------------------|-------|---------|
| 0.01%    | PointNext        | 58.4  | NeurIPS |
|          | SQN              | 45.3  | ECCV    |
|          | CPCM             | 59.3  | ICCV    |
|          | AADNet           | 60.8  | AAAI    |
|          | **DiCoSeg (Ours)**   | **61.1** |    [logs](https://drive.google.com/file/d/1JjYUofwmHeGM-Ikvm1rw37pXLAVOG2r1/view?usp=drive_link) / [ckpt](https://drive.google.com/file/d/1cjSIqpzWTNBwuuyasCxnmUHHiyJY9hmp/view?usp=drive_link) |

| Settings | Method               | mIoU  | Where   |
|----------|----------------------|-------|---------|
| 0.1%     | SQN              | 61.4  | ECCV    |
|          | CPCM             | 66.3  | ICCV    |
|          | PointCT          | 68.3  | WACV    |
|          | AADNet           | 67.2  | AAAI    |
|          | **DiCoSeg (Ours)**   | **68.3** |   [logs](https://drive.google.com/file/d/1jeGmtlH8DyY8qnrG65oR6AfHNqd3ZBuz/view?usp=drive_link) / -     |

| Settings | Method               | mIoU  | Where   |
|----------|----------------------|-------|---------|
| 1%       | RandLA-Net       | 59.8  | CVPR    |
|          | SQN              | 63.6  | ECCV    |
|          | PointCT          | 67.6  | WACV    |
|          | **DiCoSeg (Ours)**   | **68.3** |  [logs](https://drive.google.com/file/d/1PcQORLkw50AUNoT1XKdBtABkkYXGrI29/view?usp=drive_link) / -      |

## Code update — initial release

This update adds the core SparseUNet training code and an initial SemanticKITTI configuration:

- Added `SpUNet-v1m4-diff`, including the diffusion reconstruction branch and its model registration.
- Added the SemanticKITTI 0.1% configuration at `configs/semantic_kitti/semseg-spunet-v1m4-diff-0.1.py`.
- Synchronized the training and evaluation engines, dataset support, and training script; checked the shared loss and utility dependencies.
- Retained the SparseUNet baseline (`spconv_unet_v1m1_base.py`) and diffusion model (`spconv_unet_v1m4_diff.py`), and removed unrelated model implementations and model bytecode caches.
- Removed debugging breakpoints and unused code from the diffusion model, and removed the duplicate forward call during validation.

The configuration uses `DefaultSegmentorDiff` with `SpUNet-v1m4-diff`. The training objective combines weighted segmentation cross-entropy with diffusion-branch reconstruction cross-entropy; inference uses the segmentation branch only.

After preparing the CUDA dependencies and SemanticKITTI data, the training entry point is:

```sh
sh scripts/train.sh -g 4 -d semantic_kitti \
  -c semseg-spunet-v1m4-diff-0.1 \
  -n semseg-spunet-v1m4-diff-0.1
```

**Status:** this is an initial code synchronization, not the final reproducibility release. Syntax and training-flow checks have been completed, but this snapshot has not yet been validated by a full GPU training run. Some legacy configurations still refer to model implementations removed in this update.

**A follow-up update is planned** to refine feature alignment and patch selection, clarify the weak-label sampling protocol, handle training edge cases, and provide GPU-validated reproduction instructions and results.
