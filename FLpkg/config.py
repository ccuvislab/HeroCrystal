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


from detectron2.config import CfgNode as CN


def add_config(cfg):
    """
    Add config.
    """
    _C = cfg
    
    
    
    _C.FEDSET = CN()

    # ---------------------------------------------------------------------------- #
    # Fed Settings
    # ---------------------------------------------------------------------------- #
    _C.FEDSET.DATASET_LIST = ("VOC2007_citytrain","VOC2007_kitti1")
    _C.FEDSET.ROUND = 10
    _C.FEDSET.DYNAMIC = False
    _C.FEDSET.THREAD=False
    _C.FEDSET.NUM_VGG_LAYERS = 13
    _C.FEDSET.VGG_CONV3_IDX = 6
    _C.FEDSET.DYNAMIC_CLASS = None
    _C.FEDSET.TARGET_CLASS = -1
    _C.FEDSET.BACKBONE_ONLY = False
    _C.FEDSET.ALGORITHM = "fedavg"
    _C.FEDSET.MU = 0.01
    
    # ---------------------------------------------------------------------------- #
    # Multi-teacher Settings
    # ---------------------------------------------------------------------------- #
    _C.MODEL.TEACHER_PATH=['./vgg16_caffe.pth','./vgg16_caffe.pth']
    _C.MODEL.STUDENT_PATH='./vgg16_caffe.pth'
    _C.MODEL.TEACHER_TRAINER= "pt"
    _C.MODEL.STUDENT_TRAINER= "default"
   