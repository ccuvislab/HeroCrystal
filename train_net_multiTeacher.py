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


import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator



from pt.engine.FLtrainer import FLtrainer

# to register
from pt.modeling.meta_arch.rcnn import GuassianGeneralizedRCNN
from pt.modeling.proposal_generator.rpn import GuassianRPN
from pt.modeling.roi_heads.roi_heads import GuassianROIHead
import pt.data.datasets.builtin
from pt.modeling.backbone.vgg import build_vgg_backbone
from pt.modeling.anchor_generator import DifferentiableAnchorGenerator

from shutil import copyfile

from pt import add_config
from FLpkg import add_config as FL_add_config
import os

import logging
class FRCNNTrainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        if cfg.TEST.EVALUATOR == "COCOeval":
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        if cfg.TEST.EVALUATOR == "VOCeval":
            return PascalVOCDetectionEvaluator(dataset_name)
        else:
            raise ValueError("Unknown test evaluator.")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    
    add_config(cfg)    
    FL_add_config(cfg)
    print(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    copyfile(args.config_file, os.path.join(cfg.OUTPUT_DIR, 'cfg.yaml'))
    copyfile('pt/modeling/roi_heads/fast_rcnn.py', os.path.join(cfg.OUTPUT_DIR, 'fast_rcnn.py'))
    
    

    if cfg.UNSUPNET.Trainer == "pt":
        Trainer = PTrainer
#     elif cfg.UNSUPNET.Trainer == "pteval":
#         Trainer = PTrainer
    elif cfg.UNSUPNET.Trainer == "pseudo":
        Trainer = PseudoTrainer
    elif cfg.UNSUPNET.Trainer == "frcnn":
        Trainer = FRCNNTrainer
    elif cfg.UNSUPNET.Trainer == "FLtrainer":
        Trainer = FLtrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        
        if cfg.UNSUPNET.Trainer in ["pt"]:
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelStudent)
#         elif cfg.UNSUPNET.Trainer in ["pteval"]:
#             model = Trainer.build_model(cfg)            
#             DetectionCheckpointer(
#                 model, save_dir=cfg.OUTPUT_DIR
#             ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
#             res = Trainer.test(cfg, model)
        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res
    


    trainer = Trainer(cfg)
    #trainer.resume_or_load(resume=args.resume)

    return trainer.train()


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
