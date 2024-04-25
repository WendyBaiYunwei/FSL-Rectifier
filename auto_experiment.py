import os
import torch
import argparse
import subprocess
assert torch.cuda.is_available(), "\033[31m You need GPU to Train! \033[0m"
print("CPU Count is :", os.cpu_count())

paths = [
    'animals_conv4_checkpoint.pth',
    'traffic_conv4_checkpoint.pth',
    '/mnt/hdd/yw/models/feat/mini-protonet-1-shot.pth',
    '/mnt/hdd/yw/models/feat/feat-1-shot.pth',
    '/mnt/hdd/yw/models/feat/deepsets-1-shot.pth'
]

commands = [
   f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 1 --qry_expansion 1 --aug_type sim-mix-up",
   # f"python test_fsl.py --model_class ProtoNet--backbone_class ConvNet --dataset Animals --num_eval_episodes 1000 --model_path {paths[0]} --spt_expansion 1 --qry_expansion 0 --add_transform crop+rotate",
   # f"python test_fsl.py --model_class ProtoNet--backbone_class ConvNet --dataset Traffic --num_eval_episodes 1000 --model_path {paths[1]} --spt_expansion 1 --qry_expansion 0 --add_transform original",
   # "python get_embedding.py", 
   # "python TSNE.py",  
   # f"python get_buffer_mixup_imgs.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet-buffer --model_path {paths[2]}" 
]

# commands = [
#    f"python train_fsl.py --model_class ProtoNet--backbone_class ConvNet --dataset Animals --num_eval_episodes 1000 --model_path {paths[0]} --spt_expansion 1 --qry_expansion 0 --add_transform original",
#    f"python test_fsl.py --model_class ProtoNet--backbone_class ConvNet --dataset Animals --num_eval_episodes 1000 --model_path {paths[0]} --spt_expansion 1 --qry_expansion 0 --add_transform crop+rotate",
#    f"python test_fsl.py --model_class ProtoNet--backbone_class ConvNet --dataset Traffic --num_eval_episodes 1000 --model_path {paths[1]} --spt_expansion 1 --qry_expansion 0 --add_transform original",
#    "python get_embedding.py", 
#    "python TSNE.py",   
# ]

# commands = [
   # "python train_fsl.py --max_epoch 50 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset Animals --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 1 --temperature 64 --temperature2 16 --lr 0.0001 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --gpu 1 --model_path /home/yunwei/new/FSL-Rectifier/Animals-ConvNet-Pre/0.01_0.1_[75,150,300]/pretrain_Animals.pth",
   # "python train_fsl.py --max_epoch 50 --model_class ProtoNet  --use_euclidean --backbone_class Res12 --dataset Animals --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.01 --temperature 64 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu 1 --model_path /home/yunwei/new/FSL-Rectifier/Animals-Res12-Pre/0.1_0.1_[75,150,300]/pretrain_Animals.pth",
   # "python train_fsl.py --max_epoch 50 --model_class DeepSet --use_euclidean --backbone_class ConvNet --dataset Animals --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 1 --temperature 64 --temperature2 16 --lr 0.0001 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --gpu 1 --model_path /home/yunwei/new/FSL-Rectifier/Animals-ConvNet-Pre/0.01_0.1_[75,150,300]/pretrain_Animals.pth",
   # "python train_fsl.py --max_epoch 50 --model_class DeepSet --use_euclidean --backbone_class Res12 --dataset Animals --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.01 --temperature 64 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu 1 --model_path /home/yunwei/new/FSL-Rectifier/Animals-Res12-Pre/0.1_0.1_[75,150,300]/pretrain_Animals.pth",
   # "python train_fsl.py --max_epoch 50 --model_class FEAT --use_euclidean --backbone_class ConvNet --dataset Animals --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 1 --temperature 64 --temperature2 16 --lr 0.0001 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --gpu 1 --model_path /home/yunwei/new/FSL-Rectifier/Animals-ConvNet-Pre/0.01_0.1_[75,150,300]/pretrain_Animals.pth",
   # "python train_fsl.py --max_epoch 50 --model_class FEAT --use_euclidean --backbone_class Res12 --dataset Animals --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.01 --temperature 64 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu 1 --model_path /home/yunwei/new/FSL-Rectifier/Animals-Res12-Pre/0.1_0.1_[75,150,300]/pretrain_Animals.pth"
# ]


for command in commands:
   print(f"Command: {command}")

   process = subprocess.Popen(command, shell=True)
   outcode = process.wait()
   if (outcode):
      break
