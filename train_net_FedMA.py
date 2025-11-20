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

import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from pt import add_config
from pt.engine.trainer import PTrainer
from pt.engine.trainer_sourceonly import PTrainer_sourceonly

# to register
from pt.modeling.meta_arch.rcnn import GuassianGeneralizedRCNN
from pt.modeling.proposal_generator.rpn import GuassianRPN
from pt.modeling.roi_heads.roi_heads import GuassianROIHead
import pt.data.datasets.builtin
from pt.modeling.backbone.vgg import build_vgg_backbone
from pt.modeling.anchor_generator import DifferentiableAnchorGenerator

from pt.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from shutil import copyfile

from FLpkg import FedUtils
from FLpkg import add_config as FL_add_config

from FedMA.frcnn_helper import *
from FedMA.helper_cyjui import *

import os
from collections import OrderedDict
import wandb
import logging
from concurrent.futures import ThreadPoolExecutor


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(cfg)
    FL_add_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def run_client_training(i, source_data, cfg, Trainer):
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return trainer.model


def main(args):

    cfg = setup(args)

    thread_mode = cfg.FEDSET.THREAD
    # cfg.defrost()
    output_folder = cfg.OUTPUT_DIR

    copyfile(args.config_file, os.path.join(output_folder, "cfg.yaml"))

    # set client trainer
    if cfg.UNSUPNET.Trainer == "pt":
        Trainer = PTrainer
    elif cfg.UNSUPNET.Trainer == "sourceonly":
        Trainer = PTrainer_sourceonly
    else:
        raise ValueError("Trainer Name is not found.")

    model_path_list = cfg.MODEL.TEACHER_PATH
    source_dataset_list = (
        cfg.FEDSET.DATASET_LIST
    )  # ["VOC2007_citytrain1","VOC2007_kitti1"]
    teacher_trainer = cfg.MODEL.TEACHER_TRAINER

    if len(model_path_list) != len(source_dataset_list):
        raise IndexError("models number is not consistent with dataset number")
    else:
        parties = len(source_dataset_list)

    if cfg.FEDSET.DYNAMIC_CLASS is not None:
        model_list = []

        for client_id, dynamic_class_num in enumerate(cfg.FEDSET.DYNAMIC_CLASS):
            print(cfg.FEDSET.DYNAMIC)
            cfg.defrost()
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = dynamic_class_num
            cfg.freeze()

            model_list.append(
                get_trainer(teacher_trainer, cfg, model_path_list[client_id])
            )

    else:
        model_list = [
            get_trainer(teacher_trainer, cfg, model_teacher_path)
            for model_teacher_path in model_path_list
        ]

    # -------------------start FedMA-----------------
    ### single pass
    assignments_list = []
    matching_shapes = []
    num_workers = len(model_list)
    nets, model_meta_data, layer_type = init_vgg16_rcnns(num_workers)

    # for model_idx, model in enumerate(model_list):
    #     print(len(model.state_dict()['roi_heads.box_predictor.cls_score.weight']))
    #     print(model.state_dict()['backbone.vgg_block3.0.conv2.bias'].shape)
    #     print(model.state_dict()['backbone.vgg_block3.0.conv2.weight'].shape)

    NUM_VGG_LAYERS = cfg.FEDSET.NUM_VGG_LAYERS
    VGG_CONV3_IDX = cfg.FEDSET.VGG_CONV3_IDX

    ##---------------FedMA loop
    for vgg_layer_idx in range(1, NUM_VGG_LAYERS):
        model_list = [model.cpu() for model in model_list]
        vgg_weights = pdm_prepare_weights_vggs([m.backbone for m in model_list], "cpu")

        if vgg_layer_idx < VGG_CONV3_IDX:  ### fix layers before conv3.
            vgg_weights, assignments_list = BBP_MAP_trivial(
                vgg_weights, assignments_list, vgg_layer_idx
            )
            continue
            ### note that we don't retrain for freezed layers.
        else:
            vgg_weights, assignments_list = BBP_MAP_VGG(
                vgg_weights,
                assignments_list,
                matching_shapes,
                vgg_layer_idx,
                model_meta_data,
                layer_type,
                device="cpu",
            )
        calc_matched_shape = lambda weights: [w.shape for w in weights]
        backbone_dim = [
            calc_matched_shape(vgg_weights[0])[index][0]
            for index in list(range(0, len(calc_matched_shape(vgg_weights[0])), 2))
        ]

        ### substitute matched vggs into fasterRCNN. (note that the shape of vgg might differ)
        matched_vggs = [
            matched_vgg16_no_FC(calc_matched_shape(weights)) for weights in vgg_weights
        ]
        vgg_state_dicts = [
            reconst_weights_to_state_dict(w, matched)
            for (w, matched) in zip(vgg_weights, matched_vggs)
        ]

        for vgg, state_dict in zip(matched_vggs, vgg_state_dicts):
            vgg.load_state_dict(state_dict)

        #     for model_idx, model in enumerate(model_list):
        # #     print(len(model.state_dict()['roi_heads.box_predictor.cls_score.weight']))
        #         print(model.state_dict()['backbone.vgg_block3.0.conv2.bias'].shape)
        #         print(model.state_dict()['backbone.vgg_block3.0.conv2.weight'].shape)

        initial_model_list = new_fedma_model_generator(
            cfg, backbone_dim, model_list, vgg
        )
        #     for model_idx, model in enumerate(model_list):
        # #     print(len(model.state_dict()['roi_heads.box_predictor.cls_score.weight']))
        #         print(model.state_dict()['backbone.vgg_block3.0.conv2.bias'].shape)
        #         print(model.state_dict()['backbone.vgg_block3.0.conv2.weight'].shape)

        # save model weight by different clients
        for model_idx, model_after_fedma in enumerate(initial_model_list):
            model_save_name = os.path.join(
                output_folder,
                "FedMA_{}_{}.pth".format(source_dataset_list[model_idx], vgg_layer_idx),
            )
            torch.save(model_after_fedma.state_dict(), model_save_name)
        # ---------End of FedMA, starting to train-----------

        model_list = [None] * parties

        # -------[notice !!!] thread mode has not finished yet, it is still in FedAvg version----
        if thread_mode:
            # create input args
            args_names = []
            for i, source_dataset in enumerate(source_dataset_list):
                initial_model_path = os.path.join(
                    output_folder,
                    "FedMA_{}_{}.pth".format(source_dataset, vgg_layer_idx),
                )
                print("initial_model_path={}".format(initial_model_path))
                cfg_client = setup(args)
                cfg_client.defrost()
                cfg_client.MODEL.WEIGHTS = initial_model_path
                cfg_client.MODEL.DEVICE = "cuda:" + str(i)
                cfg_client.OUTPUT_DIR = os.path.join(
                    output_folder, source_dataset + "_" + str(r)
                )
                print("output subdir={}".format(cfg_client.OUTPUT_DIR))
                cfg_client.DATASETS.TRAIN_LABEL = source_dataset
                print("current source={}".format(source_dataset))
                cfg_client.freeze()
                args_names.append((i, source_dataset, cfg_client, Trainer))
            print(len(args_names))
            with ThreadPoolExecutor() as executor:
                client_models = executor.map(
                    lambda f: run_client_training(*f), args_names
                )

            model_list = list(client_models)
        # --- for FedMA please use non-thread mode now-----
        else:

            for i, source_dataset in enumerate(source_dataset_list):
                initial_model_path = os.path.join(
                    output_folder,
                    "FedMA_{}_{}.pth".format(source_dataset, vgg_layer_idx),
                )
                print("initial_model_path={}".format(initial_model_path))
                cfg = setup(args)
                cfg.defrost()
                cfg.MODEL.WEIGHTS = initial_model_path
                cfg.OUTPUT_DIR = os.path.join(
                    output_folder, source_dataset + "_" + str(vgg_layer_idx)
                )
                print("output subdir={}".format(cfg.OUTPUT_DIR))
                cfg.DATASETS.TRAIN_LABEL = source_dataset
                print("current source={}".format(source_dataset))
                cfg.BACKBONE_DIM = backbone_dim
                cfg.FEDSET.DYNAMIC = True
                if cfg.FEDSET.DYNAMIC_CLASS is not None:
                    cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg.FEDSET.DYNAMIC_CLASS[i]

                cfg.freeze()

                trainer = Trainer(cfg)
                trainer.resume_or_load(resume=False)

                # -----model freeze
                freeze_layer_fedma(trainer.model.backbone, vgg_layer_idx)

                trainer.train()
                model_list[i] = trainer.model

    # average the last model
    wk_ratio = [1] * parties
    wk_ratio = [x / parties for x in wk_ratio]

    # model to same device
    device = torch.device("cuda:0")
    for i in range(len(model_list)):
        model_list[i] = model_list[i].to(device)

    keyword = "backbone" if cfg.FEDSET.ONLY_BACKBONE else None
    avg_model = FedUtils.avgWeight(model_list, wk_ratio, keyword)

    # save model_list

    initial_model_path = os.path.join(output_folder, "FedAvg_final.pth")
    torch.save(avg_model[0].state_dict(), initial_model_path)

    # put avg model to initial_weight
    print("save avg model to {}".format(initial_model_path))

    # close wandb for healthy # waue
    if cfg.MOON.WANDB_Enable and wandb.run:
        wandb.finish()


def freeze_layer_fedma(VGG, vgg_layer_idx):
    conv_layer_list = [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
        (4, 0),
        (4, 1),
        (4, 2),
        (5, 0),
        (5, 1),
    ]
    freeze_tuple = conv_layer_list[vgg_layer_idx - 1]
    print("freeze {}".format(freeze_tuple))
    freeze_block, freeze_layer = freeze_tuple
    for idx, (stage, _) in enumerate(VGG.stages_and_names, start=1):
        if idx < freeze_block:
            for block in stage.children():
                block.freeze()
        elif idx == freeze_block:
            for block in stage.children():
                for i_idx, layer in enumerate(block.children()):
                    if i_idx <= freeze_layer:
                        for p in layer.parameters():
                            p.requires_grad = False


def new_fedma_model_generator(cfg, backbone_dim, model_list, vgg):
    # fedma feature key
    conv_index = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    fedma_vgg_key = []
    for conv_i in conv_index:
        fedma_vgg_key.append("features.{}.weight".format(conv_i))
        fedma_vgg_key.append("features.{}.bias".format(conv_i))

    # initial model structure
    if cfg.FEDSET.DYNAMIC_CLASS is not None:

        initial_backbone_list = []
        for dynamic_class_num in cfg.FEDSET.DYNAMIC_CLASS:
            cfg.defrost()
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = dynamic_class_num
            cfg.freeze()
            Trainer = PTrainer_sourceonly
            initial_backbone = Trainer.build_model(cfg, backbone_dim, False)
            # print(initial_backbone.state_dict()['backbone.vgg_block3.0.conv2.weight'].shape)
            # detectron structure model key
            detectron_vgg_key_map = []
            for key, value in initial_backbone.backbone.state_dict().items():
                detectron_vgg_key_map.append(key)
            initial_backbone_list.append(copy.deepcopy(initial_backbone))
    else:
        Trainer = PTrainer_sourceonly
        initial_backbone = Trainer.build_model(cfg, backbone_dim, False)
        # print(initial_backbone.state_dict()['backbone.vgg_block3.0.conv2.weight'].shape)

        # detectron structure model key
        detectron_vgg_key_map = []
        for key, value in initial_backbone.backbone.state_dict().items():
            detectron_vgg_key_map.append(key)
        # craete n initial models
        initial_backbone_list = [
            copy.deepcopy(initial_backbone) for i in range(len(model_list))
        ]

    for model_idx, model_initial in enumerate(model_list):
        new_fedma_dict = OrderedDict()
        # copy backbone part from fedma model
        for key_idx, key in enumerate(fedma_vgg_key):
            detectron_key = detectron_vgg_key_map[key_idx]
            fedma_weight = vgg.state_dict()[key]
            new_fedma_dict["backbone." + detectron_key] = fedma_weight
        # copy rpn part from original model
        for key, value in model_initial.state_dict().items():
            if "backbone" not in key:
                new_fedma_dict[key] = value

        initial_backbone_list[model_idx].load_state_dict(new_fedma_dict)
    return initial_backbone_list


def load_SOmodel(cfg, model_path):
    print("load source-only pt model")
    Trainer = PTrainer_sourceonly
    if cfg.FEDSET.DYNAMIC:
        fedma_model = torch.load(model_path)
        backbone_dim = FedUtils.get_backbone_shape(fedma_model)

        cfg.defrost()
        # cfg.MODEL.WEIGHTS = model_path
        cfg.BACKBONE_DIM = backbone_dim
        cfg.freeze()

        model = Trainer.build_model(cfg, cfg.BACKBONE_DIM, False)
    else:
        model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(model_path, resume=False)

    return model


def load_FRCNNmodel_cpu(cfg, model_path):
    print("load FRCNN model")
    Trainer = DefaultTrainer
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(model_path, resume=False)
    return model.model.cpu()


def load_TSmodel_cpu(cfg, model_path):
    Trainer = PTrainer
    model = Trainer.build_model(cfg)
    model_teacher = Trainer.build_model(cfg)
    ensem_ts_model = EnsembleTSModel(model_teacher, model)
    DetectionCheckpointer(ensem_ts_model).resume_or_load(model_path, resume=False)
    return ensem_ts_model.modelStudent.cpu()


def get_trainer(trainer_name, cfg, model_path):
    if trainer_name == "pt":
        return load_TSmodel_cpu(cfg, model_path)
    elif trainer_name == "sourceonly":
        return load_SOmodel(cfg, model_path)
    elif trainer_name == "default":
        return load_FRCNNmodel_cpu(cfg, model_path)
    else:
        raise ValueError("Trainer Name is not found.")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
