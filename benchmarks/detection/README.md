
## Transferring to Object Detection

We use [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to train the object detection models.


1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

1. Put dataset under "benchmarks/detection/datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.

1. Convert the pre-trained backbone weights to detectron2's format:
   ```
   cd benchmarks/detection
   WEIGHT_FILE=../../work_dirs/selfsup/densecl/densecl_coco_800ep/extracted_densecl_coco_800ep.pth
   OUTPUT_FILE=extracted_densecl_coco_800ep.pkl
   python convert-pretrain-to-detectron2.py ${WEIGHT_FILE} ${OUTPUT_FILE}
   ```  

1. Start training:
   ```
   DET_CFG=configs/pascal_voc_R_50_C4_24k_moco.yaml
   bash run.sh ${DET_CFG} ${OUTPUT_FILE} output/run_1_voc_R_50_C4_24k_densecl_coco_800ep
   ```
         
## Transferring to Semantic Segmentation

We use [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) to train the semantic segmentation models.

1. Clone and install our modified [mmsegmentation](https://github.com/WXinlong/mmsegmentation).

1. Prepare the datasets according to [dataset_prepare.md](https://github.com/WXinlong/mmsegmentation/blob/master/docs/dataset_prepare.md).

1. Modify the pretrained model path in [config](https://github.com/WXinlong/mmsegmentation/blob/32b0affd560904d275f5b11bb3bacad62450948c/configs/densecl/fcn_r50-d8.py#L5).

1. Start training (here we use Cityscapes for example):
   ```
   CONFIG_FILE=configs/densecl/fcn_r50-d8_769x769_40k_cityscapes.py
   GPUS=4
   OUTPUT_DIR=models/fcn_r50-d8_769x769_40k_cityscapes_densecl_coco_800ep
   ./tools/dist_train.sh ${CONFIG_FILE} ${GPUS} --work-dir ${OUTPUT_DIR}
   ```