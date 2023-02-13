import cv2
import os
import json
import numpy as np
from pathlib import Path
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg



def get_balloon_dicts(img_dir):
    dataset_dicts = []

    # Find all images in the folder
    img_path = Path(img_dir)
    idx = 0
    for img in img_path.glob("*.jpg"):
        if "mask" in img.stem:
            continue
        print("Found image: ", img.stem)
        mask_img = img_path / (img.stem + "_mask.jpg")

        record = {}
        height, width = cv2.imread(str(img)).shape[:2]
        mask = cv2.imread(str(mask_img))
        record["file_name"] = str(img)
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        objs = []
        obj = {
            "bbox": [0, 0, width, height],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [mask],
            "category_id": 0,
        }
        objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
        idx += 1
    return dataset_dicts

DatasetCatalog.register("oplab_dataset", lambda : get_balloon_dicts("dataset"))
MetadataCatalog.get("oplab_dataset").set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("oplab_dataset")

dataset_dicts = get_balloon_dicts("oplab_dataset")

"""
for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("Image", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
"""


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("oplab_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

