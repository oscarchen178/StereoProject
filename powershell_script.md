# Run on windows with single 4090

### Enable Cupy PTX JIT
```
$Env:CUPY_COMPILE_WITH_PTX = "1"
```
### To enable deterministic behavior
```
$Env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
```

## For training

### On TartanAir
```
python train_stereo.py `
--mixed_precision `
--batch_size 4 `
--train_dataset TartanAir `
--lr 0.0002 `
--num_steps 100 `
--image_size 480 640 `
--train_iters 5 `
--valid_iters 5 `
--shared_backbone `
--saturation_range 0.0 1.4 `
--spatial_scale -0.2 0.4 `
--name ablation_tartanair `
--temporal `
--init_thres 0.5 `
--frame_length 4 `
--noyjitter `
--context_norm none `
--device 0
```

### On KITTIraw
```
python train_stereo.py `
--pth_name new_kitti_raw `
--mixed_precision `
--batch_size 4 `
--train_dataset kitti_raw `
--lr 0.0001 `
--num_steps 100 `
--image_size 320 1024 `
--train_iters 5 `
--valid_iters 5 `
--shared_backbone `
--saturation_range 0.7 1.3 `
--spatial_scale -0.2 0.2  `
--name KITTI_RAW `
--temporal `
--init_thres 0.5 `
--frame_length 4 `
--noyjitter `
--restore_ckpt ./checkpoints/tartanair.pth `
--context_norm none `
--device 0
```

## For evaluate

### On TartanAir
```
python .\evaluate_stereo.py `
  --dataset TartanAir `
  --mixed_precision `
  --valid_iters 5 `
  --shared_backbone `
  --temporal `
  --context_norm none `
  --restore_ckpt .\checkpoints\tartanair.pth `
  --device 0
```