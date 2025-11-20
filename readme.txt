#### Installation ###

conda create -n detectron2 python=3.8
conda activate detectron2
pip install -r requirements.txt --upgrade --upgrade-strategy only-if-needed


#####################
#   單 GPU 訓練     #
#####################
### * 用來驗證 SKF2C 資料集
## 1. SKF2C 訓練 client 
python train_net_FedAvg.py --config configs/202405_multiclass/avg04_skf2c.yaml;

'''
* 執行之後會產生 output/avg01_ck2b_so/，  當中包含 (FedAvg_2.pth, VOC2007_citytrain_2/ , VOC2007_kitti5_2/ )
* 接著要修改 configs/202405_multiclass/mt04_avg_skf2c_moon.yaml， 把這三個參數帶入
*  TEACHER_PATH: ('output/avg01_ck2b_so/VOC2007_citytrain_2/model_final.pth',
*                 'output/avg01_ck2b_so/VOC2007_kitti5_2/model_final.pth')
*  STUDENT_PATH:  'output/avg01_ck2b_so/FedAvg_2.pth'
'''

## 2. SKF2C 訓練 server 
python train_net_multiTeacher.py --config configs/202405_multiclass/mt04_avg_skf2c_moon.yaml

'''
* 執行之後會產生 ./output/mt01_avg_ck2b_so_20250422/ 內的 model_final.pth 是最後的模型
'''

### * 用來驗證 CK2B 資料集
## 1. CK2B 訓練 client 
python train_net_FedAvg.py --config configs/202405_multiclass/avg03_ck2b.yaml;
## 2. CK2B 訓練 server 
python train_net_multiTeacher.py --config configs/202405_inverse_moon/mt03_inv_ck2b.yaml


#####################
#   多 GPU 訓練     #
#####################

### 2gpu  Multi-Teacher  20240611  
# @g02 一次兩行在跑
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_multiclass/avg04_skf2c.yaml; CUDA_VISIBLE_DEVICES=0,1 python train_net_multiTeacher.py  --num-gpus 2 --config configs/202405_multiclass/mt04_avg_skf2c_moon.yaml

# @g03 分段跑
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_inverse_moon/avg04_dyn_skf2c.yaml; CUDA_VISIBLE_DEVICES=0,1 python train_net_multiTeacher.py  --num-gpus 2 --config configs/202405_inverse_moon/mt04_dyn_skf2c.yaml


# @g02 一次兩行在跑
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_inverse_moon/avg04_inv_skf2c.yaml; CUDA_VISIBLE_DEVICES=0,1 python train_net_multiTeacher.py  --num-gpus 2 --config configs/202405_inverse_moon/mt04_inv_skf2c.yaml



#####################
#  其他實驗組合紀錄 #
#####################

#### Dynamic Moon Pos-Nav in Multi-Teacher  20240607

CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_inverse_moon/avg03_dyn_ck2b.yaml
#CUDA_VISIBLE_DEVICES=0,1 python train_net_multiTeacher.py  --num-gpus 2 --config configs/202405_inverse_moon/mt03_dyn_ck2b.yaml
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_inverse_moon/avg04_dyn_skf2c.yaml
#CUDA_VISIBLE_DEVICES=0,1 python train_net_multiTeacher.py  --num-gpus 2 --config configs/202405_inverse_moon/mt04_dyn_skf2c.yaml


##### Inverse Moon Pos-Nav in Multi-Teacher Average  20240517

# mt03 inverse moon [ mt03 w/ avg03 ]
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_inverse_moon/avg03_inv_ck2b.yaml
#python train_net_FedAvg.py --config configs/202405_inverse_moon/avg03_inv_ck2b.yaml
CUDA_VISIBLE_DEVICES=0,1 python train_net_multiTeacher.py  --num-gpus 2 --config configs/202405_inverse_moon/mt03_inv_ck2b.yaml

# mt04 inverse moon [ mt04 w/ avg04 ]
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_inverse_moon/avg04_inv_skf2c.yaml
#python train_net_FedAvg.py --config configs/202405_inverse_moon/avg04_inv_skf2c.yaml
CUDA_VISIBLE_DEVICES=0,1 python train_net_multiTeacher.py  --num-gpus 2 --config configs/202405_inverse_moon/mt04_inv_skf2c.yaml


######################################
##### Multi-Teacher  20240506

python train_net_multiTeacher.py   --config configs/202405_multiclass/mt07_ma_ck2b_moon.yaml
python train_net_multiTeacher.py   --config configs/202405_multiclass/mt08_ma_skf2c_moon.yaml

######################################
##### AVG   20240411

## Avg01 so ck -> b (so)
python train_net_FedAvg.py --config configs/202405_multiclass/avg01_ck2b.yaml
## Avg02 so skf -> c (so)
python train_net_FedAvg.py --config configs/202405_multiclass/avg02_skf2c.yaml
## Avg03 moon ck -> b (moon)
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_multiclass/avg03_ck2b.yaml
## Avg04 moon skf -> c (moon)
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_multiclass/avg04_skf2c.yaml

######################################
##### MA 20240411

## MA01 moon ck -> b （moon）
python train_net_FedMA.py --config configs/202405_multiclass/ma01_ck2b.yaml
## MA02 moon skf -> c (moon)
python train_net_FedMA.py --config configs/202405_multiclass/ma02_skf2c.yaml

## 20240422 補做
## MA03 so ck -> b  (so版)
python train_net_FedMA.py --config configs/202405_multiclass/ma03_ck2b.yaml 
## MA04 so skf -> c (so版)
python train_net_FedMA.py --config configs/202405_multiclass/ma04_skf2c.yaml # 
  

######################################
######################################
#####  多gpu 跑法
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/moon/FedAvg_ck2b_sourceonly.yaml 

##### Evaluation only
python train_net.py --config configs/moon/cityeval.yaml --eval-only MODEL.WEIGHTS output/FedAvg_skf2c_sourceonly_moon_20240123/FedAvg_0.pth 

#####  single class 20240331 
## FedAvg sourceonly 
python train_net_FedAvg.py --config-file configs/FedAvg/FedAvg_skf2c_multiclass.yaml
## FedAvg moon
python train_net_FedAvg.py --config-file configs/FedAvg/FedAvg_skf2c_multiclass_moon.yaml

## FedMA sourceonly
python train_net_FedMA.py --config-file configs/FedMA/ck2b_FedMA_8cla.yaml
## FedMA moon
python train_net_FedMA.py --config-file configs/FedMA/ck2b_FedMA_8cla_moon.yaml

### git used 
git pull
git checkout 202409

git add .
git status
git commit -m ""
git push upstream 202409
