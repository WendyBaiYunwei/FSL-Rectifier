import pickle
import numpy as np
from model.FUNIT.utils import loader_from_list, get_config, write_1images
from model.utils import get_augmentations
from model.trainer.helpers import (
    prepare_model
)
from model.FUNIT.trainer import FUNIT_Trainer
import argparse
import torch
from model.models.classifier import Classifier

# get 100 randomly sampled animals embeddings
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--max_epoch', type=int, default=40) # 40 for resnet, 10 for convnet
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--dataset', type=str, default='Animals', choices=['MiniImageNet', 'TieredImagenet', 'CUB', "Animals"])    
parser.add_argument('--backbone_class', type=str, default='ConvNet', choices=['ConvNet', 'Res12'])
parser.add_argument('--schedule', type=int, nargs='+', default=[75, 150, 300], help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--query', type=int, default=15)    
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--multi_gpu', type=str, default=None)
args = parser.parse_args()

# get loader
# get the embeddings in numpy
# store the embeddings
loader = loader_from_list(
    root='/home/nus/Documents/research/augment/code/FEAT/model/FUNIT/datasets/animals',
    file_list='/home/nus/Documents/research/augment/code/FEAT/model/FUNIT/datasets/animals_list_test.txt',
    batch_size=1,
    new_size=140,
    height=128,
    width=128,
    crop=True,
    num_workers=1,
    shuffle=True,
    dataset=args.dataset)

config = get_config('./picker.yaml')

train_loader = loader_from_list(
    root=config['data_folder_train'],
    file_list=config['data_list_train'],
    batch_size=config['pool_size'],
    new_size=140,
    height=128,
    width=128,
    crop=True,
    num_workers=4,
    shuffle=True,
    dataset=config['dataset'])

trainer = FUNIT_Trainer(config)
trainer.cuda()
trainer.load_ckpt('/home/nus/Documents/research/augment/code/FEAT/model/FUNIT/pretrained/animal119_gen_00100000.pt')
trainer.eval()

picker = FUNIT_Trainer(config)
picker.cuda()
picker.load_ckpt('/home/nus/Documents/research/augment/code/FEAT/outputs/picker/checkpoints/gen_100999.pt')
picker.eval()
picker = picker.model.gen

embeddings = []
labels = []
need_expansion = True
for i, data in enumerate(loader):
    img = data[0].cuda()
    label = data[1].detach().cpu()

    for mode in ['best', 'worst']:
        qry, translation, nb = trainer.model.study_picker(picker, img, train_loader, mode)
        # print(out.shape)
        write_1images((qry, nb, translation), './analysis', postfix=f'{mode}_{i}')
    if i == 10:
        exit()
