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
from pt.engine.trainer_moon import MoonTrainer

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
import wandb
import os

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


def load_TSmodel(cfg_path, model_path):
    cfg = setup(cfg_path)
    # cfg.defrost()
    # cfg.MODEL.WEIGHTS = model_path

    Trainer = PTrainer
    model = Trainer.build_model(cfg)
    model_teacher = Trainer.build_model(cfg)
    ensem_ts_model = EnsembleTSModel(model_teacher, model)

    DetectionCheckpointer(ensem_ts_model).resume_or_load(model_path, resume=False)

    return ensem_ts_model


def get_model(dataset_name, model_num):
    if model_num == "final":
        model_name = "model_final.pth"
    else:
        model_name = "model_{0:07d}.pth".format(model_num)
    model_path = os.path.join(model_folder, dataset_name, model_name)
    cfg_path = os.path.join(model_folder, dataset_name, "cfg.yaml")
    print(cfg_path)
    print(model_path)
    return load_TSmodel(cfg_path, model_path)


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
    initial_model_path = cfg.MODEL.WEIGHTS

    copyfile(args.config_file, os.path.join(output_folder, "cfg.yaml"))

    if cfg.UNSUPNET.Trainer == "pt":
        Trainer = PTrainer
    elif cfg.UNSUPNET.Trainer == "sourceonly":
        Trainer = PTrainer_sourceonly
    elif cfg.UNSUPNET.Trainer == "moon":
        Trainer = MoonTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    # ---------load initial weight
    source_dataset_list = (
        cfg.FEDSET.DATASET_LIST
    )  # ["VOC2007_citytrain1","VOC2007_kitti1"]
    parties = len(source_dataset_list)
    # target_dataset = ['VOC2007_bddval1']

    ROUND = cfg.FEDSET.ROUND

    for r in range(ROUND):
        print("=====start round {}=====".format(r))

        print("initial_model_path={}".format(initial_model_path))

        model_list = [None] * parties

        if thread_mode:
            # create input args
            args_names = []
            for i, source_dataset in enumerate(source_dataset_list):
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

            #             # get student model
            #             model_list[i] = trainer.model
            model_list = list(client_models)
        else:

            for i, source_dataset in enumerate(source_dataset_list):
                torch.cuda.empty_cache()
                cfg = setup(args)
                cfg.defrost()
                cfg.MODEL.Global_PATH = initial_model_path
                if cfg.MOON.CONTRASTIVE_Lcon_Enable:

                    cfg.MOON.ROUND = r
                    # cfg.MODEL.GLOBAL_TRAINER = cfg.UNSUPNET.Trainer
                    # cfg.MODEL.LOCAL_TRAINER = cfg.UNSUPNET.Trainer
                    # cfg.MODEL.Global_PATH = initial_model_path
                    if r > 0:
                        previous_output_folder = os.path.join(
                            output_folder, source_dataset + "_" + str(r - 1)
                        )
                        cfg.MODEL.LOCAL_PREV_PATH = os.path.join(
                            previous_output_folder, "model_final.pth"
                        )
                    else:
                        cfg.MODEL.LOCAL_PREV_PATH = (
                            initial_model_path  # won't be used, but need to be set
                        )

                else:
                    cfg.MODEL.LOCAL_PREV_PATH = (
                        initial_model_path  # won't be used, but need to be set
                    )

                cfg.MODEL.WEIGHTS = initial_model_path
                cfg.OUTPUT_DIR = os.path.join(
                    output_folder, source_dataset + "_" + str(r)
                )
                print("output subdir={}".format(cfg.OUTPUT_DIR))
                cfg.DATASETS.TRAIN_LABEL = source_dataset
                print("current source={}".format(source_dataset))

                # ---get backbone-dim
                if cfg.FEDSET.DYNAMIC:
                    fedma_model = torch.load(initial_model_path)
                    backbone_dim = FedUtils.get_backbone_shape(fedma_model)
                    cfg.BACKBONE_DIM = backbone_dim

                cfg.freeze()

                if cfg.FEDSET.DYNAMIC_CLASS is not None:

                    print(cfg.FEDSET.DYNAMIC_CLASS[i])
                    cfg.defrost()
                    cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg.FEDSET.DYNAMIC_CLASS[i]
                    cfg.freeze()

                trainer = Trainer(cfg)
                trainer.resume_or_load(resume=False)
                trainer.train()
                model_list[i] = trainer.model

        # all done and average them

        wk_ratio = [1] * parties
        wk_ratio = [x / parties for x in wk_ratio]

        # model to same device
        device = torch.device("cuda:0")
        for i in range(len(model_list)):
            model_list[i] = model_list[i].to(device)

        keyword = "backbone" if cfg.FEDSET.BACKBONE_ONLY else None

        avg_model = FedUtils.avgWeight(model_list, wk_ratio, keyword)

        # save model_list

        initial_model_path = os.path.join(output_folder, "FedAvg_" + str(r) + ".pth")
        torch.save(avg_model[0].state_dict(), initial_model_path)

        # put avg model to initial_weight
        print("save avg model to {}".format(initial_model_path))
    # close wandb for healthy # waue
    if cfg.MOON.WANDB_Enable and wandb.run:
        wandb.finish()


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
