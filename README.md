# SDoM: A diagnostic model for staging pneumoconiosis based on data expansion and KL entropy judgement


<p align="center">
  <img src="https://github.com/nonoXwb/SDom/tree/master/assets/final_pipeline.png">
</p>

### Pneumoconiosis X-Ray Chest X-ray Dataset
We split Pneumoconiosis X-Ray Chest X-ray Dataset to simplify the input of annotations, we generate [train list]and [test list]. Each line is composed of the image name and the corresponding labels like below:
```
00000001_002.png 0 1 1 0 0 0 0 0 0 0 0 0 0 0
```
If the image is positive with one class, the corresponding bit is 1, otherwise is 0. 

## Unsupervised Lung Swapping Pre-training
The command is following. Please fill in the blanks with your own paths.
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
  --master_port=8898 train_ARMGan.py \
  --size 256 \
  --batch 8 \
  --lr 0.001 \
  --trlist data/trainval_list.txt \
  --tslist data/test_list.txt \
  --wandb \
  --proj_name lsae \
  [XRC_PATH] [XRC_Mask_PATH]
```

### Generate analysis of results
<p align="center">
  <img src="https://github.com/nonoXwb/SDom/tree/master/assets/final_teaser.png" width="720">
</p>

## Training MKTransformer
The command is following. Please fill in the blanks with your own paths. Before running, you need to download the [pretrained_lsae.pt](https://drive.google.com/file/d/1Qh-BhnAQIdnvO7bd--RArIOtzobwR9FQ/view?usp=sharing), and put it in the directory *saved_ckpts*.
```
CUDA_VISIBLE_DEVICES=0 python train_MKTransformer.py \
  --path [XCR_PATH] \
  --batch 96 \
  --iter 35000 \
  --lr 0.01 \
  --lr_steps 26000 30000 \
  --trlist data/priori_list.txt \
  --tslist data/test_list.txt \
  --enc_ckpt saved_ckpts/pretrained_MKTransformer.pt \
  --wandb
```

