import os
import torch
import argparse
import subprocess
assert torch.cuda.is_available(), "\033[31m You need GPU to Train! \033[0m"
print("CPU Count is :", os.cpu_count())


# 		python train_fsl.py utils yaml model_path
# 		utils: model_class, use_euclidean, backbone_class, dataset
# num_eval_episodes
# experiment table for 4 expansions for both traffic and animals

paths = [
    '/home/yunwei/new/FSL-Rectifier/Animals-ConvNet-Pre/\'0.01_0.1_[75, 150, 300]\'/checkpoint.pth',
    '/home/yunwei/new/FSL-Rectifier/Animals-Res12-Pre/\'0.1_0.1_[75, 150, 300]\'/checkpoint.pth',
    '/home/yunwei/new/FSL-Rectifier/Traffic-ConvNet-Pre/\'0.01_0.1_[75, 150, 300]\'/checkpoint.pth',
    '/home/yunwei/new/FSL-Rectifier/Traffic-Res12-Pre/\'0.1_0.1_[75, 150, 300]\'/checkpoint.pth'
]
# commands = [
#    f"python train_fsl.py --model_class ProtoNet--backbone_class ConvNet --dataset Animals --num_eval_episodes 1000 --model_path {paths[0]}",
#    f"python train_fsl.py --model_class ProtoNet--backbone_class ConvNet --dataset Animals --num_eval_episodes 50 --use_euclidean --model_path {paths[0]}",
#    f"python train_fsl.py --model_class ProtoNet--backbone_class Res12 --dataset Animals --num_eval_episodes 1000 --model_path {paths[1]}",
#    f"python train_fsl.py --model_class ProtoNet--backbone_class Res12 --dataset Animals --num_eval_episodes 50 --use_euclidean --model_path {paths[1]}",
#    f"python train_fsl.py --model_class ProtoNet--backbone_class ConvNet --dataset Traffic --num_eval_episodes 1000 --model_path {paths[2]}",
#    f"python train_fsl.py --model_class ProtoNet--backbone_class ConvNet --dataset Traffic --num_eval_episodes 50 --use_euclidean --model_path {paths[2]}",
#    f"python train_fsl.py --model_class ProtoNet--backbone_class Res12 --dataset Traffic --num_eval_episodes 1000 --model_path {paths[3]}",
#    f"python train_fsl.py --model_class ProtoNet--backbone_class Res12 --dataset Traffic --num_eval_episodes 50 --use_euclidean --model_path {paths[3]}",
# ]

commands = [
   f"python train_fsl_traffic.py --model_class ProtoNet --backbone_class ConvNet --dataset Traffic --num_eval_episodes 10 --model_path {paths[2]}  --add_transform crop+rotate",
   # f"python train_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset Animals --num_eval_episodes 1 --model_path {paths[0]}",
   # f"python train_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset Animals --num_eval_episodes 10 --use_euclidean --model_path {paths[0]} --add_transform crop+rotate",
   # f"python train_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset Animals --num_eval_episodes 20 --model_path {paths[1]} --add_transform crop+rotate",
   # f"python train_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset Animals --num_eval_episodes 10 --use_euclidean --model_path {paths[1]} --add_transform crop+rotate",
   # f"python train_fsl_traffic.py --model_class ProtoNet --backbone_class ConvNet --dataset Traffic --num_eval_episodes 50 --use_euclidean --model_path {paths[2]}",
   # f"python train_fsl_traffic.py --model_class ProtoNet --backbone_class Res12 --dataset Traffic --num_eval_episodes 100 --model_path {paths[3]}",
   # f"python train_fsl_traffic.py --model_class ProtoNet --backbone_class Res12 --dataset Traffic --num_eval_episodes 50 --use_euclidean --model_path {paths[3]}",
]


for command in commands:
   print(f"Command: {command}")

   process = subprocess.Popen(command, shell=True)
   outcode = process.wait()
   if (outcode):
      break
