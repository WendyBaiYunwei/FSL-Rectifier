import os
import torch
import argparse
import subprocess
assert torch.cuda.is_available(), "\033[31m You need GPU to Train! \033[0m"
print("CPU Count is :", os.cpu_count())

paths = [
   # animals conv feat 0
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Animals-FEAT-ConvNet-05w01s15q-SIM/20_0.2_lr0.0001mul10_step_T11T21_b0_bsz080-NoAug/epoch-last.pth',
   # animals res12 feat 1
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Animals-FEAT-Res12-05w01s15q-DIS/40_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz080-NoAug/epoch-last.pth', 
   '/mnt/hdd/yw/models/feat/mini-protonet-1-shot.pth', #2 mini proto
   '/mnt/hdd/yw/models/feat/feat-1-shot.pth', #3 cub feat
   '/mnt/hdd/yw/models/feat/deepsets-1-shot.pth', #4 cub deepsets
   # animal conv proto 5
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Animals-ProtoNet-ConvNet-05w01s15q-DIS/20_0.5_lr0.0001mul10_step_T164.0T216.0_b1.0_bsz080-NoAug/epoch-last.pth',
   # cub proto 6
   '/mnt/hdd/yw/models/feat/cub-proto-1shot.pth',
   # animals res proto 7
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Animals-ProtoNet-Res12-05w01s15q-DIS/40_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz080-NoAug/epoch-last.pth',
   # animals conv deepset 8
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Animals-DeepSet-ConvNet-05w01s15q-SIM/20_0.5_lr0.01mul10_step_T11T21_b0_bsz080-NoAug/epoch-last.pth',
   # animals res deepset 9
   '/home/yunwei/new/FSL-Rectifier/checkpoints/Animals-DeepSet-Res12-05w01s15q-DIS/40_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz080-NoAug/epoch-last.pth',
   # cub feat 10
   '/mnt/hdd/yw/models/feat/cub-feat-1shot.pth',
   # mini deepset 11
   '/mnt/hdd/yw/models/feat/mini-deepsets-1-shot.pth', 
   # mini feat 12
   '/mnt/hdd/yw/models/feat/mini-feat-1-shot.pth', 
]

table1 = [
   # first column
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 1 --eval_shot 2 --eval_query 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 1 --aug_type crop+rotate --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 1 --aug_type color --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 1 --aug_type affine --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 1 --aug_type image_translator --num_eval_episodes 5000",
   
   # res 12
   f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[7]} --spt_expansion 1 --qry_expansion 1 --eval_shot 2 --eval_query 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 5000",
   f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[7]} --spt_expansion 1 --qry_expansion 1 --aug_type crop+rotate --num_eval_episodes 5000",
   f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[7]} --spt_expansion 1 --qry_expansion 1 --aug_type color --num_eval_episodes 5000",
   f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[7]} --spt_expansion 1 --qry_expansion 1 --aug_type affine --num_eval_episodes 5000",
   f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[7]} --spt_expansion 1 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 5000",
   f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[7]} --spt_expansion 1 --qry_expansion 1 --aug_type image_translator --num_eval_episodes 5000",

   # second column
   f"python test_fsl.py --model_class DeepSet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[8]} --spt_expansion 1 --qry_expansion 1 --eval_shot 2 --eval_query 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 5000",
   f"python test_fsl.py --model_class DeepSet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[8]} --spt_expansion 1 --qry_expansion 1 --aug_type crop+rotate --num_eval_episodes 5000",
   f"python test_fsl.py --model_class DeepSet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[8]} --spt_expansion 1 --qry_expansion 1 --aug_type color --num_eval_episodes 5000",
   f"python test_fsl.py --model_class DeepSet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[8]} --spt_expansion 1 --qry_expansion 1 --aug_type affine --num_eval_episodes 5000",
   f"python test_fsl.py --model_class DeepSet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[8]} --spt_expansion 1 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 5000",
   f"python test_fsl.py --model_class DeepSet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[8]} --spt_expansion 1 --qry_expansion 1 --aug_type image_translator --num_eval_episodes 5000",
   
   # res 12
   f"python test_fsl.py --model_class DeepSet --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[9]} --spt_expansion 1 --qry_expansion 1 --eval_shot 2 --eval_query 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 5000",
   f"python test_fsl.py --model_class DeepSet --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[9]} --spt_expansion 1 --qry_expansion 1 --aug_type crop+rotate --num_eval_episodes 5000",
   f"python test_fsl.py --model_class DeepSet --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[9]} --spt_expansion 1 --qry_expansion 1 --aug_type color --num_eval_episodes 5000",
   f"python test_fsl.py --model_class DeepSet --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[9]} --spt_expansion 1 --qry_expansion 1 --aug_type affine --num_eval_episodes 5000",
   f"python test_fsl.py --model_class DeepSet --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[9]} --spt_expansion 1 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 5000",
   f"python test_fsl.py --model_class DeepSet --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[9]} --spt_expansion 1 --qry_expansion 1 --aug_type image_translator --num_eval_episodes 5000",

   # third column
   f"python test_fsl.py --model_class FEAT --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[0]} --spt_expansion 1 --qry_expansion 1 --eval_shot 2 --eval_query 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 5000",
   f"python test_fsl.py --model_class FEAT --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[0]} --spt_expansion 1 --qry_expansion 1 --aug_type crop+rotate --num_eval_episodes 5000",
   f"python test_fsl.py --model_class FEAT --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[0]} --spt_expansion 1 --qry_expansion 1 --aug_type color --num_eval_episodes 5000",
   f"python test_fsl.py --model_class FEAT --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[0]} --spt_expansion 1 --qry_expansion 1 --aug_type affine --num_eval_episodes 5000",
   f"python test_fsl.py --model_class FEAT --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[0]} --spt_expansion 1 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 5000",
   f"python test_fsl.py --model_class FEAT --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[0]} --spt_expansion 1 --qry_expansion 1 --aug_type image_translator --num_eval_episodes 5000",
   
   # res 12
   f"python test_fsl.py --model_class FEAT --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[1]} --spt_expansion 1 --qry_expansion 1 --eval_shot 2 --eval_query 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 5000",
   f"python test_fsl.py --model_class FEAT --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[1]} --spt_expansion 1 --qry_expansion 1 --aug_type crop+rotate --num_eval_episodes 5000",
   f"python test_fsl.py --model_class FEAT --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[1]} --spt_expansion 1 --qry_expansion 1 --aug_type color --num_eval_episodes 5000",
   f"python test_fsl.py --model_class FEAT --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[1]} --spt_expansion 1 --qry_expansion 1 --aug_type affine --num_eval_episodes 5000",
   f"python test_fsl.py --model_class FEAT --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[1]} --spt_expansion 1 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 5000",
   f"python test_fsl.py --model_class FEAT --backbone_class Res12 --dataset animals --use_euclidean --model_path {paths[1]} --spt_expansion 1 --qry_expansion 1 --aug_type image_translator --num_eval_episodes 5000",
]

table2 = [
   # cub protonet
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 1 --qry_expansion 1 --eval_shot 2 --eval_query 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 1 --qry_expansion 1 --aug_type crop+rotate --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 1 --qry_expansion 1 --aug_type color --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 1 --qry_expansion 1 --aug_type affine --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 1 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 5000",
   
   # # cub feat
   # f"python test_fsl.py --model_class FEAT --backbone_class ConvNet --dataset cub --model_path {paths[10]} --spt_expansion 1 --qry_expansion 1 --eval_shot 2 --eval_query 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class FEAT --backbone_class ConvNet --dataset cub --model_path {paths[10]} --spt_expansion 1 --qry_expansion 1 --aug_type crop+rotate --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class FEAT --backbone_class ConvNet --dataset cub --model_path {paths[10]} --spt_expansion 1 --qry_expansion 1 --aug_type color --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class FEAT --backbone_class ConvNet --dataset cub --model_path {paths[10]} --spt_expansion 1 --qry_expansion 1 --aug_type affine --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class FEAT --backbone_class ConvNet --dataset cub --model_path {paths[10]} --spt_expansion 1 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 5000",
   
   # # mini proto
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 1 --qry_expansion 1 --eval_shot 2 --eval_query 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 1 --qry_expansion 1 --aug_type crop+rotate --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 1 --qry_expansion 1 --aug_type color --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 1 --qry_expansion 1 --aug_type affine --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 1 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 5000",
   
   # # mini deepset
   # f"python test_fsl.py --model_class DeepSet --backbone_class Res12 --dataset miniImagenet --model_path {paths[11]} --spt_expansion 1 --qry_expansion 1 --eval_shot 2 --eval_query 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class DeepSet --backbone_class Res12 --dataset miniImagenet --model_path {paths[11]} --spt_expansion 1 --qry_expansion 1 --aug_type crop+rotate --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class DeepSet --backbone_class Res12 --dataset miniImagenet --model_path {paths[11]} --spt_expansion 1 --qry_expansion 1 --aug_type color --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class DeepSet --backbone_class Res12 --dataset miniImagenet --model_path {paths[11]} --spt_expansion 1 --qry_expansion 1 --aug_type affine --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class DeepSet --backbone_class Res12 --dataset miniImagenet --model_path {paths[11]} --spt_expansion 1 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 5000",
   
   # # mini feat
   # f"python test_fsl.py --model_class FEAT --backbone_class Res12 --dataset miniImagenet --model_path {paths[12]} --spt_expansion 1 --qry_expansion 1 --eval_shot 2 --eval_query 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class FEAT --backbone_class Res12 --dataset miniImagenet --model_path {paths[12]} --spt_expansion 1 --qry_expansion 1 --aug_type crop+rotate --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class FEAT --backbone_class Res12 --dataset miniImagenet --model_path {paths[12]} --spt_expansion 1 --qry_expansion 1 --aug_type color --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class FEAT --backbone_class Res12 --dataset miniImagenet --model_path {paths[12]} --spt_expansion 1 --qry_expansion 1 --aug_type affine --num_eval_episodes 5000",
   # f"python test_fsl.py --model_class FEAT --backbone_class Res12 --dataset miniImagenet --model_path {paths[12]} --spt_expansion 1 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 5000",
]

table3 = [
   # animals, 3 expansions
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 0 --qry_expansion 1 --aug_type image_translator --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 0 --qry_expansion 2 --aug_type image_translator --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 0 --qry_expansion 3 --aug_type image_translator --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 0 --aug_type image_translator --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 1 --aug_type image_translator --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 2 --aug_type image_translator --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 3 --aug_type image_translator --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 2 --qry_expansion 0 --aug_type image_translator --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 2 --qry_expansion 1 --aug_type image_translator --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 2 --qry_expansion 2 --aug_type image_translator --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 2 --qry_expansion 3 --aug_type image_translator --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 3 --qry_expansion 0 --aug_type image_translator --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 3 --qry_expansion 1 --aug_type image_translator --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 3 --qry_expansion 2 --aug_type image_translator --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 3 --qry_expansion 3 --aug_type image_translator --num_eval_episodes 500",

   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 0 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 0 --qry_expansion 2 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 0 --qry_expansion 3 --aug_type sim-mix-up --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 0 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 2 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 3 --aug_type sim-mix-up --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 2 --qry_expansion 0 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 2 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 2 --qry_expansion 2 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 2 --qry_expansion 3 --aug_type sim-mix-up --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 3 --qry_expansion 0 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 3 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 3 --qry_expansion 2 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 3 --qry_expansion 3 --aug_type sim-mix-up --num_eval_episodes 500",

   # cub
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 0 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 0 --qry_expansion 2 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 0 --qry_expansion 3 --aug_type sim-mix-up --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 1 --qry_expansion 0 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 1 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 1 --qry_expansion 2 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 1 --qry_expansion 3 --aug_type sim-mix-up --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 2 --qry_expansion 0 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 2 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 2 --qry_expansion 2 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 2 --qry_expansion 3 --aug_type sim-mix-up --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 3 --qry_expansion 0 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 3 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 3 --qry_expansion 2 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 3 --qry_expansion 3 --aug_type sim-mix-up --num_eval_episodes 500",

   # # mini
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 0 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 0 --qry_expansion 2 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 0 --qry_expansion 3 --aug_type sim-mix-up --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 1 --qry_expansion 0 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 1 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 1 --qry_expansion 2 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 1 --qry_expansion 3 --aug_type sim-mix-up --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 2 --qry_expansion 0 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 2 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 2 --qry_expansion 2 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 2 --qry_expansion 3 --aug_type sim-mix-up --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 3 --qry_expansion 0 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 3 --qry_expansion 1 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 3 --qry_expansion 2 --aug_type sim-mix-up --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 3 --qry_expansion 3 --aug_type sim-mix-up --num_eval_episodes 500",

   # cub
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 0 --qry_expansion 1 --eval_shot 1 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 0 --qry_expansion 2 --eval_shot 1 --eval_query 3 --aug_type true-test --num_eval_episodes 500",
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 0 --qry_expansion 3 --eval_shot 1 --eval_query 4 --aug_type true-test --num_eval_episodes 500",
   
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 1 --qry_expansion 0 --eval_shot 2 --eval_query 1 --aug_type true-test --num_eval_episodes 500",
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 1 --qry_expansion 1 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 1 --qry_expansion 2 --eval_shot 2 --eval_query 3 --aug_type true-test --num_eval_episodes 500",
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 1 --qry_expansion 3 --eval_shot 2 --eval_query 4 --aug_type true-test --num_eval_episodes 500",
   
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 2 --qry_expansion 0 --eval_shot 3 --eval_query 1 --aug_type true-test --num_eval_episodes 500",
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 2 --qry_expansion 1 --eval_shot 3 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 2 --qry_expansion 2 --eval_shot 3 --eval_query 3 --aug_type true-test --num_eval_episodes 500",
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 2 --qry_expansion 3 --eval_shot 3 --eval_query 4 --aug_type true-test --num_eval_episodes 500",
   
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 3 --qry_expansion 0 --eval_shot 4 --eval_query 1 --aug_type true-test --num_eval_episodes 500",
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 3 --qry_expansion 1 --eval_shot 4 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 3 --qry_expansion 2 --eval_shot 4 --eval_query 3 --aug_type true-test --num_eval_episodes 500",
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 3 --qry_expansion 3 --eval_shot 4 --eval_query 4 --aug_type true-test --num_eval_episodes 500",

   # mini
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 0 --qry_expansion 1 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 0 --qry_expansion 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 0 --qry_expansion 3 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 1 --qry_expansion 0 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 1 --qry_expansion 1 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 1 --qry_expansion 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 1 --qry_expansion 3 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 2 --qry_expansion 0 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 2 --qry_expansion 1 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 2 --qry_expansion 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 2 --qry_expansion 3 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 3 --qry_expansion 0 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 3 --qry_expansion 1 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 3 --qry_expansion 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class Res12 --dataset miniImagenet --model_path {paths[2]} --spt_expansion 3 --qry_expansion 3 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",

   # # animals
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 0 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 1 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 3 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 2 --qry_expansion 0 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 2 --qry_expansion 1 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 2 --qry_expansion 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 2 --qry_expansion 3 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 3 --qry_expansion 0 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 3 --qry_expansion 1 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 3 --qry_expansion 2 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 3 --qry_expansion 3 --eval_shot 2 --eval_query 2 --aug_type true-test --num_eval_episodes 500",

]

# training = [
   # "python train_fsl.py --max_epoch 50 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset Animals --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 1 --temperature 64 --temperature2 16 --lr 0.0001 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --gpu 1 --model_path /home/yunwei/new/FSL-Rectifier/Animals-ConvNet-Pre/0.01_0.1_[75,150,300]/pretrain_Animals.pth",
   # "python train_fsl.py --max_epoch 50 --model_class ProtoNet  --use_euclidean --backbone_class Res12 --dataset Animals --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.01 --temperature 64 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu 1 --model_path /home/yunwei/new/FSL-Rectifier/Animals-Res12-Pre/0.1_0.1_[75,150,300]/pretrain_Animals.pth",
   # "python train_fsl.py --max_epoch 50 --model_class DeepSet --use_euclidean --backbone_class ConvNet --dataset Animals --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 1 --temperature 64 --temperature2 16 --lr 0.0001 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --gpu 1 --model_path /home/yunwei/new/FSL-Rectifier/Animals-ConvNet-Pre/0.01_0.1_[75,150,300]/pretrain_Animals.pth",
   # "python train_fsl.py --max_epoch 50 --model_class DeepSet --use_euclidean --backbone_class Res12 --dataset Animals --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.01 --temperature 64 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu 1 --model_path /home/yunwei/new/FSL-Rectifier/Animals-Res12-Pre/0.1_0.1_[75,150,300]/pretrain_Animals.pth",
   # "python train_fsl.py --max_epoch 50 --model_class FEAT --use_euclidean --backbone_class ConvNet --dataset Animals --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 1 --temperature 64 --temperature2 16 --lr 0.0001 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --gpu 1 --model_path /home/yunwei/new/FSL-Rectifier/Animals-ConvNet-Pre/0.01_0.1_[75,150,300]/pretrain_Animals.pth",
   # "python train_fsl.py --max_epoch 50 --model_class FEAT --use_euclidean --backbone_class Res12 --dataset Animals --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.01 --temperature 64 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu 1 --model_path /home/yunwei/new/FSL-Rectifier/Animals-Res12-Pre/0.1_0.1_[75,150,300]/pretrain_Animals.pth"
# ]

ablation = [
   # random mix-up or image translation
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 1 --aug_type random-mix-up --num_eval_episodes 40", # inverse
   f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 1 --aug_type random-image_translator --num_eval_episodes 40",
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 1 --aug_type random-mix-up --num_eval_episodes 40", 
   # f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[5]} --spt_expansion 1 --qry_expansion 1 --aug_type random-image_translator --num_eval_episodes 40",
]
single = [f"python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset cub --model_path {paths[6]} --spt_expansion 1 --qry_expansion 1 --eval_query 2 --eval_shot 2 --aug_type true-test --num_eval_episodes 500"]

commands = single
for command in commands:
   print(f"Command: {command}")

   process = subprocess.Popen(command, shell=True)
   outcode = process.wait()
   if (outcode):
      break
