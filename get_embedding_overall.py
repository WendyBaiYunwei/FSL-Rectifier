import pickle
import numpy as np
from model.utils import get_augmentations
from model.trainer.helpers import (
    prepare_model
)
from model.image_translator.trainer import Translator_Trainer
from model.image_translator.utils import (
    get_recon, get_trans, get_sim,
    loader_from_list, get_config
)
import argparse
import torch
from model.models.protonet import ProtoNet
import torchvision.transforms.functional as TF

sample_iters = 10000
AUGMENT = False
# path = '/home/yunwei/new/rf/feat/checkpoints/MiniImageNet-ProtoNet-ConvNet-05w01s15q-Pre-DIS/20_0.5_lr0.0001mul10_step_T164.0T216.0_b1.0_bsz080-NoAug/max_acc.pth'
path = '/mnt/hdd/yw/models/feat/mini-protonet-1-shot.pth'
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--init_weights', type=str, default=None)  
parser.add_argument('--multi_gpu', type=str, default=None)  
parser.add_argument('--model_class', type=str, default='ProtoNet')  
parser.add_argument('--dataset', type=str, default='miniImagenet')    
parser.add_argument('--backbone_class', type=str, default='Res12', choices=['ConvNet', 'Res12'])
parser.add_argument('--query', type=int, default=1)    
parser.add_argument('--model_path', type=str, default=path)
args = parser.parse_args()

mean = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]  # Mean values for RGB channels
std = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]

config = get_config('miniImagenet.yaml')
test_loader = loader_from_list(
    root=config['data_folder_test'],
    file_list=config['data_list_test'],
    batch_size=32,
    new_size=84,
    height=84,
    width=84,
    crop=True,
    num_workers=1,
    return_paths=True,
    dataset=args.dataset)

train_loader = loader_from_list(
    root=config['data_folder_train'],
    file_list=config['data_list_train'],
    batch_size=64,
    new_size=84,
    height=84,
    width=84,
    crop=True,
    num_workers=4,
    return_paths=True,
    dataset=args.dataset)

model, _ = prepare_model(args)
# model = model.cuda()
embeddings = []
all_embeddings = []
old_embeddings = []
labels = []

loader = test_loader
for i, data in enumerate(loader):
    if i%100 == 0:
        print(i)
    img = data[0].cuda()
    label = data[1].detach().cpu()
    paths = data[-1]
    keep_idx = torch.logical_and(10 < label, label < 18)

    img = img[keep_idx]
    if len(img) == 0:
        continue
    label = label[keep_idx]
    if i >= sample_iters:
        break
    
    augimgs1 = []
    augimgs2 = []
    augimgs3 = []
    for i, path in enumerate(paths):
        if keep_idx[i] == False:
            continue
        augimg = get_sim(path, expansion_size=3, dataset='miniImagenet')
        augimgs1.append(augimg[0])
    #     augimgs2.append(augimg[1])
    #     augimgs3.append(augimg[2])
    augimgs1 = torch.stack(augimgs1).squeeze()
    # augimgs2 = torch.stack(augimgs2).squeeze()
    # augimgs3 = torch.stack(augimgs3).squeeze()
    
    img = TF.normalize(img, mean=mean, std=std)
    augimgs1 = TF.normalize(augimgs1, mean=mean, std=std)
    # augimgs2 = TF.normalize(augimgs2, mean=mean, std=std)
    # augimgs3 = TF.normalize(augimgs3, mean=mean, std=std)
    

    # img = TF.normalize(img, mean=mean, std=std)
    # img = torch.cat([augimg, img]).mean(0).unsqueeze(0)
    # img = img.repeat(80, 1, 1, 1)
    

    # print(reconstructed_img.shape)
    # exit()
    # reconstructed_img = img # for training / testing
    old_embedding = model(img, get_feature=True)
    embedding = model(augimgs1, get_feature=True)


    # embedding2 = model(augimgs1, get_feature=True)
    # embedding3 = model(augimgs2, get_feature=True)
    # embedding4 = model(augimgs3, get_feature=True)
    # embedding = 0.25 * (old_embedding + embedding2 + embedding3 + embedding4)

    # if AUGMENT == True:
    #     class_expansions = get_trans(config, path, expansion_size=AUGMENTATION_SIZE)
    #     aug_embeddings = model(class_expansions, get_feature=True)
    #     embedding = torch.cat([embedding, aug_embeddings]).mean(0)
    old_embeddings.append(old_embedding.detach().cpu())
    all_embeddings.append(embedding.detach().cpu())
    labels.append(label)

all_embeddings = np.concatenate(all_embeddings).reshape(-1, 640)
old_embeddings = np.concatenate(old_embeddings).reshape(-1, 640)
print(all_embeddings.shape)
labels = np.concatenate(labels).reshape(-1, 1)
print(labels.shape)

with open('embeddings_old.pkl', 'wb') as f:
    pickle.dump(old_embeddings, f)

with open('embeddings_sim1.pkl', 'wb') as f:
    pickle.dump(all_embeddings, f)

with open('embeddings_sim1_labels.pkl', 'wb') as f:
    pickle.dump(labels, f)
