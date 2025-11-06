#create gt
#python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos cfgs/dataset_configs/kitti_dataset.yaml

#train-car
#python train.py --cfg_file ./cfgs/kitti_models/voxel_rcnn_car_focal_multimodal.yaml  \
#--batch_size 4 --epochs 50 --num_epochs_to_eval 15

#train-car-per  #2-3d metric=0: bbox, 1: bev, 2: 3d ; m=类别；l=难度 ; k=阈值
#CUDA_VISIBLE_DEVICES='0' python train.py --cfg_file ./cfgs/kitti_models/voxel_rcnn_car_focal_multimodal.yaml \
#--batch_size 2 --epochs 80 --workers 1 --num_epochs_to_eval 25 --max_ckpt_save_num 25 \
##--pretrained_model ../output/cfgs/kitti_models/voxel_rcnn_car_focal_multimodal/default/ckpt/24.4.10-ep47-88.11/checkpoint_epoch_47.pth
#
#CUDA_VISIBLE_DEVICES='0' python test.py --cfg_file ./cfgs/kitti_models/voxel_rcnn_car_focal_multimodal_test.yaml  --batch_size 1 \
#--workers 1 --eval_all

#test-car
CUDA_VISIBLE_DEVICES="0" python test.py --cfg_file ./cfgs/kitti_models/voxel_rcnn_car_focal_multimodal_test.yaml  --batch_size 1 \
--ckpt ../output/cfgs/kitti_models/voxel_rcnn_car_focal_multimodal/default/ckpt/DATA/24.4.15-ep49-87.4/checkpoint_epoch_49.pth \
--save_to_file

#python test.py --cfg_file ./cfgs/kitti_models/voxel_rcnn_car_focal_multimodal_test.yaml  --batch_size 1 \
#--ckpt ../output/cfgs/kitti_models/voxel_rcnn_car_focal_multimodal/default/ckpt/25.7.1-ep77-88.12-right/checkpoint_epoch_77.pth \

#test-3class
#python test.py --cfg_file ./cfgs/kitti_models/voxel_rcnn_3class_focal.yaml  --batch_size 1 \
#--ckpt ../output/cfgs/kitti_models/voxel_rcnn_3class_focal/default/ckpt/checkpoint_epoch_73.pth \
#--save_to_file

#demo
#python  demo.py --cfg_file ./cfgs/kitti_models/voxel_rcnn_car_focal_multimodal.yaml \
#  --ckpt ../output/cfgs/kitti_models/voxel_rcnn_car_focal_multimodal/default/ckpt/25.6.25-ep74-88.24-full/checkpoint_epoch_74.pth \
#  --data_path /home/fanyuling/FYL-code/OpenPCDet/data/kitti/training/velodyne/000000.bin

#CUDA_VISIBLE_DEVICES="1"

##train-3class-pointpillars  #2-3d metric=0: bbox, 1: bev, 2: 3d ; m=类别；l=难度 ; k=阈值
#CUDA_VISIBLE_DEVICES='0' python train.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml \
#--batch_size 4 --epochs 80 --workers 1 --num_epochs_to_eval 30 --max_ckpt_save_num 30 \
#
#CUDA_VISIBLE_DEVICES='0' python test.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml  --batch_size 1 \
#--workers 1 --eval_all

#CUDA_VISIBLE_DEVICES='0' python train.py --cfg_file ./cfgs/kitti_models/voxel_rcnn_car.yaml \
#--batch_size 2 --epochs 80 --workers 1 --num_epochs_to_eval 30 --max_ckpt_save_num 30 \
#
#CUDA_VISIBLE_DEVICES='0' python test.py --cfg_file ./cfgs/kitti_models/voxel_rcnn_car.yaml  --batch_size 1 \
#--workers 1 --eval_all

#CUDA_VISIBLE_DEVICES="1" python test.py --cfg_file ./cfgs/kitti_models/voxel_rcnn_car.yaml  --batch_size 1 \
#--ckpt ../output/cfgs/kitti_models/voxel_rcnn_car/default/ckpt/25.9.2-ep80-85.20-ep75-56.13-ep69-77.14/checkpoint_epoch_75.pth \