# -*- coding: utf-8 -*-
import os
import torch
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import structures 
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader
from detectron2.structures.boxes import Boxes
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer


from pt.modeling.proposal_generator.rpn import GuassianRPN
from pt.modeling.roi_heads.roi_heads import GuassianROIHead
import pt.data.datasets.builtin
from pt.engine.trainer import PTrainer
from pt.engine.trainer_sourceonly import PTrainer_sourceonly
from pt.modeling.meta_arch.rcnn import GuassianGeneralizedRCNN
from pt.modeling.backbone.vgg import build_vgg_backbone
from pt.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from pt import add_config
from functools import wraps
from FLpkg import add_config as FL_add_config


##----------config
def setup_all(config_file):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(cfg)
    FL_add_config(cfg)
    cfg.merge_from_file(config_file)

    cfg.freeze()
    return cfg

def setup(config_file):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(config_file)
    #cfg.merge_from_list(args.opts)
    
    #default_setup(cfg, args)
#     cfg.SOLVER.IMG_PER_BATCH_LABEL = 64
#     cfg.SOLVER.IMG_PER_BATCH_UNLABEL = 64

    cfg.freeze()
    return cfg




def scaling(proposal_roih,ratio):
    scale_pred = proposal_roih.get('pred_boxes')
    scale_pred.scale(ratio,ratio)
    return scale_pred
##------------model
def get_model(func):
    @wraps(func)
    def warp(dataset_name,model_num):        
        
        model_folder = 'keep_experiments'

        if model_num =='final':
            model_name ='model_final.pth'
        else:
            model_name = 'model_{0:07d}.pth'.format(model_num)
        model_path = os.path.join(model_folder,dataset_name,model_name)
        cfg_path = os.path.join(model_folder,dataset_name,'cfg.yaml')
        print(cfg_path)
        print(model_path)
        return func(cfg_path, model_path)
    return warp



@get_model
def load_TSmodel(cfg_path, model_path):
    cfg = setup(cfg_path)
    #cfg.defrost()
    #cfg.MODEL.WEIGHTS = model_path
    
    Trainer =PTrainer
    model = Trainer.build_model(cfg)
    model_teacher = Trainer.build_model(cfg)
    ensem_ts_model = EnsembleTSModel(model_teacher, model)    
    
    
    DetectionCheckpointer(ensem_ts_model).resume_or_load(model_path, resume=False)
    
    return ensem_ts_model



def load_sourceonlyModel(cfg_path,model_path):
    cfg = setup(cfg_path)   
    
    Trainer =PTrainer_sourceonly
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(model_path, resume=False)
    
    return model 


def load_FRCNNmodel(cfg, model_path):
    #cfg = setup(cfg_path)
    #cfg.defrost()
    #cfg.MODEL.WEIGHTS = model_path
    
    Trainer =DefaultTrainer
    model = Trainer.build_model(cfg)    
    
    DetectionCheckpointer(model).resume_or_load(model_path, resume=False)
    
    return model


# #-----------prediction



def get_proposal_roih(data,model):

    with torch.no_grad():
        (_,  proposals_rpn_unsup_k, proposals_roih_unsup_k, _,) =model.modelStudent(
            data, branch="unsup_data_weak")
    return proposals_roih_unsup_k

##------------pseudo labeling
def get_match_array(pairwise_iou_results):
    match_array=[None]*len(pairwise_iou_results)
    for i in range(len(pairwise_iou_results)):
        if torch.sum(pairwise_iou_results[i])==0:
            match_array[i] = False
            #print("{} no match index".format(i))
        else:
            match_array[i] = True
    return match_array

def get_match_array_all(proposals_roih, gt,ratio):
    source_num = len(proposals_roih)
    #proposals_roih: array
    #gt = data_annotation[0]['annotations']
    
    
    box_list=[]
    for ann in gt:
        box_list.append(ann['bbox'])
    bboxes_gt = structures.Boxes(torch.Tensor(box_list)).to("cuda")
    
    source_prediction = []
    for i, proposals_roih_n in enumerate(proposals_roih):  
        
        prediction_temp = scaling(proposals_roih_n,ratio)
        source_prediction.append(prediction_temp)
        
    #pairwise_gt_list = []
    match_array_gt =[]
    for source_prediction_n in source_prediction:
        pairwise_gt_src = structures.pairwise_iou(bboxes_gt,source_prediction_n)
        match_gt_src = get_match_array(pairwise_gt_src)
        #print(pairwise_gt_src)
        #pairwise_gt_list.append(pairwise_gt_src)
        match_array_gt.append(match_gt_src)
    
    #pairwise_src2others = []
    match_array_source = []
    for i, source_prediction_n in enumerate(source_prediction):
        match_array_source_n =[]
        #gt
        source_n_match_gt = structures.pairwise_iou(source_prediction_n,bboxes_gt)
        #pairwise_src2gt.append(source_n_match_gt)
        match_array_source_n.append(get_match_array(source_n_match_gt))
        # others
        #pairwise_sa_sb = []        
        for j in range(source_num):
            if j!=i:
                sourcen_n_match_other = structures.pairwise_iou(source_prediction_n,source_prediction[j])
                #pairwise_sa_sb.append(sourcen_n_match_other)
                match_array_source_n.append(get_match_array(sourcen_n_match_other))
        match_array_source.append(match_array_source_n)                                                            
                                                              
        
    return  match_array_gt, match_array_source



def get_TP(ma_gt,source_list):
    
    
    ma_gt_array = np.array(ma_gt).T
    df_gt = pd.DataFrame(ma_gt_array, columns = source_list)

    source_num = len(source_list)
    TP=0
    TN_array = [None]*source_num
    df_gt['summary'] = df_gt.sum(axis=1)    
    TP = sum(df_gt.summary == source_num)
    for i in range(source_num):
        TN_array[i] = sum(df_gt.summary == i)
    return TP, TN_array  


def get_FP(ma_src,source_list):
    source_num = len(source_list)
    df_src=[None]*len(source_list)
    for i, source_list_name in enumerate(source_list):
        col_name = [ x for k,x in enumerate(source_list) if k!=i]
        col_name.insert(0, 'groundtruth')
        df_src[i] = pd.DataFrame()    
        src_array = np.array(ma_src[i]).T

        df_src[i] = pd.DataFrame(src_array, columns = col_name)
#=============================        
    FP_array_all = []
    for j in range(source_num):
        # sum except gt
        df=df_src[j]
        df['summary'] = df.drop('groundtruth', axis=1).sum(axis=1)
        # drop gt=true (not false positive)
        df = df[df.groundtruth == 0]
        FP_array = [None]*source_num
        for i in range(source_num):        
            FP_array[i] = sum(df.summary == i)
        FP_array_all.append(FP_array)
    return FP_array_all
##---------------visualize

def drawbb(image_filename, target_metadata, bboxes_to_draw, output_name):
    im = cv2.imread(image_filename, cv2.IMREAD_COLOR)[:, :, ::-1]
    v = Visualizer(
            im[:, :, ::-1], 
            metadata=target_metadata, 
            scale=1,
            )
    for box in bboxes_to_draw.to('cpu'):
        v.draw_box(box,edge_color='b')
        #v.draw_text(str(box[:2].numpy()), tuple(box[:2].numpy()),color='b')

    v = v.get_output()
    img =  v.get_image()[:, :, ::-1]
    cv2.imwrite(output_name, img)


def drawimage_bb_text(image_tensor, bboxes_to_draw, output_name):#,classes, scores,drawcolor='b'):
    image_array = image_tensor.permute(1, 2, 0).cpu().numpy()

    # Create a Visualizer instance
    v = Visualizer(image_array)

    for i, box in enumerate(bboxes_to_draw.to('cpu')):
        #scores_numpy = scores[i].to('cpu').numpy()
        #if scores_numpy>0.1:
        v.draw_box(box,edge_color='b')
            #v.draw_text(str(classes[i].to('cpu').numpy())+" " +str(scores_numpy), tuple(box[:2].numpy()) ,color=drawcolor)

    v = v.get_output()
    img =  v.get_image()[:, :, ::-1]
    cv2.imwrite(output_name, img)    


def drawbb_text(image_filename, target_metadata, bboxes_to_draw, output_name, classes, scores, drawcolor='b'):
    im = cv2.imread(image_filename, cv2.IMREAD_COLOR)[:, :, ::-1]
    v = Visualizer(
            im[:, :, ::-1], 
            metadata=target_metadata, 
            scale=1,
            )
    for i, box in enumerate(bboxes_to_draw.to('cpu')):
        if scores is not None and scores[i] is not None:
            score = scores[i].to('cpu').numpy()
            v.draw_box(box, edge_color=drawcolor)
            v.draw_text(str(classes[i].to('cpu').numpy()) + " " + str(score), tuple(box[:2].numpy()), color=drawcolor)
        else:
            v.draw_box(box, edge_color=drawcolor)
            v.draw_text(str(classes[i].to('cpu').numpy()), tuple(box[:2].numpy()), color=drawcolor)

    v = v.get_output()
    img =  v.get_image()[:, :, ::-1]
    cv2.imwrite(output_name, img)

    


def eval_metric_summary(test_data_loader,data_annotation, model_list, source_list,ratio, output_file ):
    source_num = len(source_list)
    
    f = open(output_file, "w")
    
    column_name = "filename,TP,"
    for i in range(source_num):
        column_name+="FN"+str(i)+","
    for i in range(source_num):
        column_name+="FP"+str(i)+","
    column_name = column_name[:-1]
    f.write(column_name+"\n")

    count=0
    
    for idx,test_data in enumerate(test_data_loader):    
#         if count>5:
#             break
        
        file_name = os.path.basename(test_data[0]['file_name'])
        #print(file_name)
        
        proposals_roih_multiple =[]
        for model in model_list:
            proposals_roih_multiple.append(get_proposal_roih(test_data,model))        
        
        ma_gt, ma_src = get_match_array_all(proposals_roih_multiple,data_annotation[idx]['annotations'] , ratio)
        #print(ma_gt)
        TP, TN_array  =get_TP(ma_gt,source_list)
        #TP, TN1, TN2
        FP_array = get_FP(ma_src,source_list)

        result =file_name+","+str(TP)+","
        for i,TN in enumerate(TN_array):
            result+=str(TN)+","   
        #print(FP_array)
        FP_sum = np.array(FP_array).sum(axis=0)
        #print(FP_sum)
        for i,FP in enumerate(FP_sum):
            result+=str(FP)+","
        result=result[:-1]
        f.write(result+"\n")
        
        
        count+=1
        
    f.close()

    

def gen_pie_chart_list(eval_ck2b):
    pie_chart_list = []
    for index in eval_ck2b.columns:
        if 'FP' in index:
            pie_chart_list.append(eval_ck2b.sum(axis=0)[index]/3)    
        elif 'filename' not in index:
            pie_chart_list.append(eval_ck2b.sum(axis=0)[index])
    return pie_chart_list


def draw_pie_chart(pie_chart_list, labels,DA_dataset, detector):
#labels =['TN_s1','TN_s2','TN_3','TP','FP_s1','FP_s2','FP_3']
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect="equal"))
    wedges, texts, autotexts =ax.pie(pie_chart_list,
            labels=labels,
            #radius=1.5,
            textprops={'color':'black',  'size':12},  # 設定文字樣式
            pctdistance=0.8,
            autopct=lambda i: f'{i:.1f}%' ,
            wedgeprops={'linewidth':3,'edgecolor':'w'})   # 繪製每個扇形的外框
    percentage = np.round([a for a in pie_chart_list]/(sum(pie_chart_list)*np.ones(len(pie_chart_list)))*100,2)

    labels = [f'{l}, {s:0.1f}%' for l, s in zip(labels, percentage)]
    
    ax.legend(   labels,
          loc="center left",
          bbox_to_anchor=(1.5, 0, 0.5, 1))
    plt.setp(autotexts, size=12,color='white', weight="bold")
    ax.set_title("FP/FN analysis, {} dataset using {}".format(DA_dataset, detector),size=18)
    plt.show()
