# 🚧 DiCoSeg

🎯 **Diffusion-Based Contextual Reconstruction for Point Cloud Segmentation with Limited Annotations** 

🔬 Tackling 3D segmentation under annotation scarcity? DiCoSeg leverages diffusion models for **context-aware reconstruction** with just a few labels.  
⚡️ Single-step inference, strong generalization, and robust results across benchmarks.

📈 Currently under submission — stay tuned for updates!  
📬 *If you're reviewing this... be kind 😄*

📎 [Code coming soon]

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
|          | **DiCoSeg (Ours)**   | **68.3** |         |

| Settings | Method               | mIoU  | Where   |
|----------|----------------------|-------|---------|
| 1%       | RandLA-Net       | 59.8  | CVPR    |
|          | SQN              | 63.6  | ECCV    |
|          | PointCT          | 67.6  | WACV    |
|          | **DiCoSeg (Ours)**   | **68.3** |         |

