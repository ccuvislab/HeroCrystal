#### Dataset ####

可以放在/work底下並在/FedCoin/Data下用link方式取用資料集，避免資料集太多導致空間不足


#####################
#     2GPU訓練      #
#####################

*** CK2B ***

## 1. CK2B 訓練 client
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_multiclass/avg03_ck2b.yaml

**執行之後，假設output資料夾 為 output/avg01_ck2b_so/， 則該資料夾中包含 (FedAvg_2.pth, VOC2007_citytrain_2/ , VOC2007_kitti5_2/ )

## 2. CK2B 訓練 server
CUDA_VISIBLE_DEVICES=0,1 python train_net_multiTeacher.py --num-gpus 2 --config configs/202405_multiclass/mt03_avg_ck2b_moon.yaml

*** SKF2C ***

## 1. SKF2C 訓練 client
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_multiclass/avg04_skf2c.yaml

## 2. SKF2C 訓練 server
CUDA_VISIBLE_DEVICES=0,1 python train_net_multiTeacher.py --num-gpus 2 --config configs/202405_multiclass/mt04_avg_skf2c_moon.yaml

#####################
#     其他訓練      #
#####################

---FedProx---

## CK2B 訓練 client
CUDA_VISIBLE_DEVICES=0,1 python train_net_unified.py --num-gpus 2 --config configs/202405_multiclass/prox01_ck2b.yaml


==== 備註 ====

*** config修改 ***

## 1. 跑多個訓練時，主要修改DATASET_LIST下使用的link，不用去改動DATASET_LIST的內容

## 2. 修改OUTPUT_DIR和WANDB_Project_Name(若有使用)

## 3. MultiTeacher訓練之config需多修改TEACHER_PATH和STUDENT_PATH

