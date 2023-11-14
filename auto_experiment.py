import os
import torch
import argparse
import subprocess
assert torch.cuda.is_available(), "\033[31m You need GPU to Train! \033[0m"
print("CPU Count is :", os.cpu_count())


# 		python test_fsl.py utils yaml model_path
# 		utils: model_class, use_euclidean, backbone_class, dataset
# num_eval_episodes
# experiment table for 4 expansions for both traffic and animals

paths = [
    '/home/yunwei/new/FSL-Rectifier/Animals-ConvNet-Pre/\'0.01_0.1_[75, 150, 300]\'/checkpoint.pth',
    '/home/yunwei/new/FSL-Rectifier/Animals-Res12-Pre/\'0.1_0.1_[75, 150, 300]\'/checkpoint.pth',
    '/home/yunwei/new/FSL-Rectifier/Traffic-ConvNet-Pre/\'0.01_0.1_[75, 150, 300]\'/checkpoint.pth',
    '/home/yunwei/new/FSL-Rectifier/Traffic-Res12-Pre/\'0.1_0.1_[75, 150, 300]\'/checkpoint.pth'
]

fsl_paths = [
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Animals-ProtoNet-ConvNet-05w01s15q-SIM/20_0.5_lr0.01mul10_step_T11T21_b0_bsz080-NoAug/epoch-last.pth',
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Animals-ProtoNet-ConvNet-05w01s15q-DIS/20_0.5_lr0.01mul10_step_T11T21_b0_bsz080-NoAug/epoch-last.pth',
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Animals-FEAT-ConvNet-05w01s15q-SIM/20_0.5_lr0.01mul10_step_T11T21_b0_bsz080-NoAug/epoch-last.pth',
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Animals-FEAT-ConvNet-05w01s15q-DIS/20_0.5_lr0.01mul10_step_T11T21_b0_bsz080-NoAug/epoch-last.pth',
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Animals-DeepSet-ConvNet-05w01s15q-SIM/20_0.5_lr0.01mul10_step_T11T21_b0_bsz080-NoAug/epoch-last.pth',
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Animals-DeepSet-ConvNet-05w01s15q-DIS/20_0.5_lr0.01mul10_step_T11T21_b0_bsz080-NoAug/epoch-last.pth',
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Traffic-ProtoNet-ConvNet-05w01s15q-SIM/20_0.5_lr0.01mul10_step_T11T21_b0_bsz080-NoAug/epoch-last.pth',
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Traffic-ProtoNet-ConvNet-05w01s15q-DIS/20_0.5_lr0.01mul10_step_T11T21_b0_bsz080-NoAug/epoch-last.pth',
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Traffic-FEAT-ConvNet-05w01s15q-SIM/20_0.5_lr0.01mul10_step_T11T21_b0_bsz080-NoAug/epoch-last.pth',
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Traffic-FEAT-ConvNet-05w01s15q-DIS/20_0.5_lr0.01mul10_step_T11T21_b0_bsz080-NoAug/epoch-last.pth',
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Traffic-DeepSet-ConvNet-05w01s15q-SIM/20_0.5_lr0.01mul10_step_T11T21_b0_bsz080-NoAug/epoch-last.pth',
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Traffic-DeepSet-ConvNet-05w01s15q-DIS/20_0.5_lr0.01mul10_step_T11T21_b0_bsz080-NoAug/epoch-last.pth',
]
# commands = [
#    f"python test_fsl.py --model_class ProtoNet--backbone_class ConvNet --dataset Animals --num_eval_episodes 1000 --model_path {paths[0]}",
#    f"python test_fsl.py --model_class ProtoNet--backbone_class ConvNet --dataset Animals --num_eval_episodes 50 --use_euclidean --model_path {paths[0]}",
#    f"python test_fsl.py --model_class ProtoNet--backbone_class Res12 --dataset Animals --num_eval_episodes 1000 --model_path {paths[1]}",
#    f"python test_fsl.py --model_class ProtoNet--backbone_class Res12 --dataset Animals --num_eval_episodes 50 --use_euclidean --model_path {paths[1]}",
#    f"python test_fsl.py --model_class ProtoNet--backbone_class ConvNet --dataset Traffic --num_eval_episodes 1000 --model_path {paths[2]}",
#    f"python test_fsl.py --model_class ProtoNet--backbone_class ConvNet --dataset Traffic --num_eval_episodes 50 --use_euclidean --model_path {paths[2]}",
#    f"python test_fsl.py --model_class ProtoNet--backbone_class Res12 --dataset Traffic --num_eval_episodes 1000 --model_path {paths[3]}",
#    f"python test_fsl.py --model_class ProtoNet--backbone_class Res12 --dataset Traffic --num_eval_episodes 50 --use_euclidean --model_path {paths[3]}",
# ]

commands = [
   "python pretrain.py --max_epoch 40 --lr 0.01 --dataset Animals --backbone_class ConvNet",
   # ProtoNet
   f"python train_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset Animals --max_epoch 40 --episodes_per_epoch 100 --model_path {paths[0]} --lr 0.01 --gamma 0.5", 
   f"python train_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset Animals --max_epoch 40 --episodes_per_epoch 100 --model_path {paths[0]} --lr 0.01 --gamma 0.5 --use_euclidean", 
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset Animals --num_eval_episodes 50 --model_path {fsl_paths[0]} --add_transform crop+rotate",
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset Animals --num_eval_episodes 50 --use_euclidean --model_path {fsl_paths[1]} --add_transform crop+rotate",
   
   f"python train_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset Traffic --max_epoch 30 --episodes_per_epoch 100 --model_path {paths[2]} --lr 0.01 --gamma 0.5", 
   f"python train_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset Traffic --max_epoch 30 --episodes_per_epoch 100 --model_path {paths[2]} --lr 0.01 --gamma 0.5 --use_euclidean", 
   f"python test_fsl_traffic.py --model_class ProtoNet --backbone_class ConvNet --dataset Traffic --num_eval_episodes 50 --model_path {fsl_paths[6]} --add_transform original",
   f"python test_fsl_traffic.py --model_class ProtoNet --backbone_class ConvNet --dataset Traffic --num_eval_episodes 50 --use_euclidean --model_path {fsl_paths[7]} --add_transform original",

   # FEAT
   f"python train_fsl.py --model_class FEAT --backbone_class ConvNet --dataset Animals --max_epoch 50 --episodes_per_epoch 100 --model_path {paths[0]} --lr 0.01 --gamma 0.5", 
   f"python train_fsl.py --model_class FEAT --backbone_class ConvNet --dataset Animals --max_epoch 50 --episodes_per_epoch 100 --model_path {paths[0]} --lr 0.01 --gamma 0.5 --use_euclidean", 
   f"python test_fsl.py --model_class FEAT --backbone_class ConvNet --dataset Animals --num_eval_episodes 50 --model_path {fsl_paths[2]} --add_transform crop+rotate",
   f"python test_fsl.py --model_class FEAT --backbone_class ConvNet --dataset Animals --num_eval_episodes 50 --use_euclidean --model_path {fsl_paths[3]} --add_transform crop+rotate",

   f"python train_fsl.py --model_class FEAT --backbone_class ConvNet --dataset Traffic --max_epoch 30 --episodes_per_epoch 100 --model_path {paths[2]} --lr 0.01 --gamma 0.5", 
   f"python train_fsl.py --model_class FEAT --backbone_class ConvNet --dataset Traffic --max_epoch 30 --episodes_per_epoch 100 --model_path {paths[2]} --lr 0.01 --gamma 0.5 --use_euclidean", 
   f"python test_fsl_traffic.py --model_class FEAT --backbone_class ConvNet --dataset Traffic --num_eval_episodes 50 --model_path {fsl_paths[8]}  --add_transform original",
   f"python test_fsl_traffic.py --model_class FEAT --backbone_class ConvNet --dataset Traffic --num_eval_episodes 50 --use_euclidean --model_path {fsl_paths[9]} --add_transform original",

   # f"python train_fsl.py --model_class DeepSet --backbone_class ConvNet --dataset Traffic --max_epoch 30 --episodes_per_epoch 100 --model_path {paths[2]} --lr 0.01 --gamma 0.5", 
   # f"python train_fsl.py --model_class DeepSet --backbone_class ConvNet --dataset Traffic --max_epoch 30 --episodes_per_epoch 100 --model_path {paths[2]} --lr 0.01 --gamma 0.5 --use_euclidean", 

   # f"python train_fsl.py --model_class DeepSet --backbone_class ConvNet --dataset Animals --max_epoch 200 --episodes_per_epoch 100 --model_path {paths[0]} --lr 0.01 --gamma 0.5", 
   # f"python train_fsl.py --model_class DeepSet --backbone_class ConvNet --dataset Animals --max_epoch 200 --episodes_per_epoch 100 --model_path {paths[0]} --lr 0.01 --gamma 0.5 --use_euclidean", 



   # f"python test_fsl_traffic.py --model_class DeepSet --backbone_class ConvNet --dataset Traffic --num_eval_episodes 10 --model_path {fsl_paths[10]}  --add_transform original",
   # f"python test_fsl_traffic.py --model_class DeepSet --backbone_class ConvNet --dataset Traffic --num_eval_episodes 10 --use_euclidean --model_path {fsl_paths[11]} --add_transform original",
   # f"python test_fsl.py --model_class DeepSet --backbone_class ConvNet --dataset Animals --num_eval_episodes 10 --model_path {fsl_paths[4]} --add_transform crop+rotate",
   # f"python test_fsl.py --model_class DeepSet --backbone_class ConvNet --dataset Animals --num_eval_episodes 10 --use_euclidean --model_path {fsl_paths[5]} --add_transform crop+rotate",

   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset Animals --num_eval_episodes 50 --model_path {fsl_paths[1]} --add_transform crop+rotate",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset Animals --num_eval_episodes 50 --use_euclidean --model_path {fsl_paths[1]} --add_transform crop+rotate",
   # f"python test_fsl_traffic.py --model_class ProtoNet --backbone_class Res12 --dataset Traffic --num_eval_episodes 50 --model_path {fsl_paths[3]} --add_transform original",
   # f"python test_fsl_traffic.py --model_class ProtoNet --backbone_class Res12 --dataset Traffic --num_eval_episodes 10 --use_euclidean --model_path {fsl_paths[3]} --add_transform original",
]


for command in commands:
   print(f"Command: {command}")

   process = subprocess.Popen(command, shell=True)
   outcode = process.wait()
   if (outcode):
      break
