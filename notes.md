##dataset

##training
protonet, deepsets, feat, res12/conv4, euclidean distance/cosine sim, animals/traffic
protonet, conv4, pretrain, euclidean distance, animals, parameters

python train_fsl.py  --max_epoch 100 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset Animals --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 1 --temperature 64 --temperature2 16 --lr 0.0001 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --gpu 1 --init_weights /home/yunwei/new/FSL-Rectifier/Animals-ConvNet-Pre/0.01_0.1_[75, 150, 300]/pretrain_Animals.pth --eval_interval 20


protonet, res12, pretrain, euclidean distance, animals, parameters

python train_fsl.py  --max_epoch 200 --model_class ProtoNet  --backbone_class Res12 --dataset Animals --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.01 --temperature 64 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu 1 --init_weights /home/yunwei/new/FSL-Rectifier/Animals-Res12-Pre/0.1_0.1_[75, 150, 300]/pretrain_Animals.pth --eval_interval 20 --use_euclidean

deepsets, conv4, pretrain, euclidean distance, animals, parameters

python train_fsl.py  --max_epoch 100 --model_class DeepSets --use_euclidean --backbone_class ConvNet --dataset Animals --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 1 --temperature 64 --temperature2 16 --lr 0.0001 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --gpu 1 --init_weights /home/yunwei/new/FSL-Rectifier/Animals-ConvNet-Pre/0.01_0.1_[75, 150, 300]/pretrain_Animals.pth --eval_interval 20

deepsets, res12, pretrain, euclidean distance, animals, parameters

python train_fsl.py  --max_epoch 200 --model_class DeepSets  --backbone_class Res12 --dataset Animals --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.01 --temperature 64 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu 1 --init_weights /home/yunwei/new/FSL-Rectifier/Animals-Res12-Pre/0.1_0.1_[75, 150, 300]/pretrain_Animals.pth --eval_interval 20 --use_euclidean
