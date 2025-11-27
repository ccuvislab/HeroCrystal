# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import contextlib
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
import io
import logging
from detectron2.data.datasets.pascal_voc import register_pascal_voc

logger = logging.getLogger(__name__)

JSON_ANNOTATIONS_DIR = ""
_SPLITS_COCO_FORMAT = {}
_SPLITS_COCO_FORMAT["coco"] = {
    "coco_2017_unlabel": (
        "coco/unlabeled2017",
        "coco/annotations/image_info_unlabeled2017.json",
    ),
    "coco_2017_for_voc20": (
        "coco",
        "coco/annotations/google/instances_unlabeledtrainval20class.json",
    ),
}


def register_coco_unlabel(root):
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_unlabel_instances(
                key, meta, os.path.join(root, json_file), os.path.join(root, image_root)
            )


def register_coco_unlabel_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_unlabel_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_coco_unlabel_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))

    dataset_dicts = []

    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        dataset_dicts.append(record)

    return dataset_dicts


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("VOC2007_citytrain", 'data/VOC2007_citytrain', "train", 8),
        ("VOC2007_cityval", 'data/VOC2007_cityval', "val", 8),
        ("VOC2007_citytrain1", 'data/VOC2007_citytrain', "train", 1),
        ("VOC2007_citytrainsmall", 'data/VOC2007_citytrain', "train_small", 1),

        ("VOC2007_citytrain5", 'data/VOC2007_citytrain', "train", 5),
        ("VOC2007_cityval1", 'data/VOC2007_cityval', "val", 1),
        ("VOC2007_cityval5", 'data/VOC2007_cityval', "val", 5),
        ("VOC2007_cityvalsmall", 'data/VOC2007_cityval', "val_small", 1),
        
        ("VOC2007_foggytrain", 'data/VOC2007_foggytrain', "train", 8),
        ("VOC2007_foggyval", 'data/VOC2007_foggyval', "val", 8),
        ("VOC2007_foggytrain1", 'data/VOC2007_foggytrain', "train", 1),
        ("VOC2007_foggytrainsmall", 'data/VOC2007_foggytrain', "train_small", 1),

        ("VOC2007_foggyval1", 'data/VOC2007_foggyval', "val", 1),
       
        ("VOC2007_bddtrain8", 'data/VOC2007_bddtrain', "train", 8),
        ("VOC2007_bddval8", 'data/VOC2007_bddval', "val", 8),
        ("VOC2007_bddval8small", 'data/VOC2007_bddval', "val_small", 8),
        ("VOC2007_bddtrain1", 'data/VOC2007_bddtrain', "train", 1),
        ("VOC2007_bddval1", 'data/VOC2007_bddval', "val", 1),
        ("VOC2007_bddvalsmall", 'data/VOC2007_bddval', "val_small", 1),
        ("VOC2007_kitti1", 'data/kitti', "train", 1),
        ("VOC2007_kittismall", 'data/kitti', "train_small", 1),
        ("VOC2007_kitti5", 'data/kitti', "train", 5),
        ("VOC2007_kitti8", 'data/kitti', "train", 8),
        ("VOC2007_sim1", 'data/sim', "train", 1),
        ("VOC2007_simsmall", 'data/sim', "train_small", 1),
        
        ("multi-ck", 'data/multi-ck', "train", 1),
        ("multi-skf", 'data/multi-skf', "train", 1),
        
        #ck2b_bddfake#
        ("VOC2007_citytrain_bddfake", 'data/VOC2007_citytrain_bddfake', "train", 8),
        ("VOC2007_kitti5_bddfake", 'data/kitti_bddfake', "train", 5),
        ("VOC2007_bddval8_bddfake", 'data/VOC2007_bddval_bddfake', "val", 8),

        ("VOC2007_bddtrain8_bddfake", 'data/VOC2007_bddtrain_bddfake', "train", 1),
        
        #ck2b_bddfake_rare#
        ("VOC2007_citytrain_bddfake_rare", 'data/VOC2007_citytrain_bddfake_rare', "train", 8),
        ("VOC2007_kitti5_bddfake_rare", 'data/kitti_bddfake_rare', "train", 5),
        ("VOC2007_bddval8_bddfake_rare", 'data/VOC2007_bddval_bddfake_rare', "val", 8),

        ("VOC2007_bddtrain8_bddfake_rare", 'data/VOC2007_bddtrain_bddfake_rare', "train", 1),
        
        #skf2c#
        ("VOC2007_foggytrain_skf", 'data/VOC2007_foggytrain_skf', "train", 8),
        ("VOC2007_sim1_skf", 'data/sim_skf', "train", 1),
        ("VOC2007_kitti5_skf", 'data/kitti_skf', "train", 5),
        ("VOC2007_cityval_skf", 'data/VOC2007_cityval_skf', "val", 8),
        
        ("VOC2007_citytrain_skf", 'data/VOC2007_citytrain_skf', "train", 8),
        ("VOC2007_bddval8small_skf", 'data/VOC2007_bddval_skf', "val_small", 8),
        
        #skf2c_cityfake#
        ("VOC2007_foggytrain_cityfake", 'data/VOC2007_foggytrain_cityfake', "train", 8),
        ("VOC2007_sim1_cityfake", 'data/sim_cityfake', "train", 1),
        ("VOC2007_kitti5_cityfake", 'data/kitti_cityfake', "train", 5),
        ("VOC2007_cityval_cityfake", 'data/VOC2007_cityval_cityfake', "val", 8),
        
        ("VOC2007_citytrain_cityfake", 'data/VOC2007_citytrain_cityfake', "train", 8),
        ("VOC2007_bddval8small_cityfake", 'data/VOC2007_bddval_cityfake', "val_small", 8),
        
        #skf2c_cityfake_rare#
        ("VOC2007_foggytrain_cityfake_rare", 'data/VOC2007_foggytrain_cityfake_rare', "train", 8),
        ("VOC2007_sim1_cityfake_rare", 'data/sim_cityfake_rare', "train", 1),
        ("VOC2007_kitti5_cityfake_rare", 'data/kitti_cityfake_rare', "train", 5),
        ("VOC2007_cityval_cityfake_rare", 'data/VOC2007_cityval_cityfake_rare', "val", 8),
        
        ("VOC2007_citytrain_cityfake_rare", 'data/VOC2007_citytrain_cityfake_rare', "train", 8),
        ("VOC2007_bddval8small_cityfake_rare", 'data/VOC2007_bddval_cityfake_rare', "val_small", 8),

    ]
    
            
    for name, dirname, split, cls in SPLITS:
        year = 2012
        if cls == 1:
            class_names = ('car',)
        elif cls == 20:
            class_names = (
                "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            )
        elif cls == 8:
            class_names = ('car','truck',  'rider', 'person', 'train', 'motorcycle', 'bicycle', 'bus')
        elif cls == 7:
            class_names = ('car','truck',  'rider', 'person', 'motorcycle', 'bicycle', 'bus')
        elif cls == 5:
            class_names = ('car','truck',  'rider', 'person', 'train')
        else:
            raise RuntimeError
        register_pascal_voc(name, os.path.join(root, dirname), split, year, class_names)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


_root = os.getenv("DETECTRON2_DATASETS", "")
register_coco_unlabel(_root)
register_all_pascal_voc(_root)
