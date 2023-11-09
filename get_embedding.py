import pickle
import numpy as np
from model.FUNIT.utils import loader_from_list, get_config, get_augmentations
from model.trainer.helpers import (
    prepare_model
)
from model.FUNIT.trainer import FUNIT_Trainer
import argparse
# get 100 randomly sampled animals embeddings
sample_iters = 2000
path = '/home/nus/Documents/research/augment/code/FEAT/Animals-ConvNet-Pre/0.01_0.1_[75, 150, 300]/last.pth'
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
parser.add_argument('--model_class', type=str, default='ProtoNet')
parser.add_argument('--init_weights', type=str, default=path)
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
    num_workers=1)
model, _ = prepare_model(args)
config = get_config('./picker.yaml')
trainer = FUNIT_Trainer(config)
# loaded_dict = torch.load(path)['params']
# new_params = {}
# for k in model.state_dict().keys():
#     new_params[k] = loaded_dict[k]
# model.load_state_dict(new_params)
model = model.cuda()
embeddings = []
labels = []
expansion_size = 5
for i, data in enumerate(loader):
    if i%100 == 0:
        print(i)
    img = data[0].cuda()
    label = data[1].detach().cpu()
    if i >= sample_iters: ## sample_iters
        break
    reconstructed_img = trainer.model.pick_animals(img, expansion_size=0, random=args.random_picker)
    embedding = model(reconstructed_img, get_feature=True)
    embeddings.append(embedding.detach().cpu())
    labels.append(label)
    if label == 0: # get expansion
        expansion = get_augmentations(img, expansion_size, 'color', get_img=True)
        embedding = model(expansion, get_feature=True)
        embeddings.append(embedding.detach().cpu())
        labels.append(995)

        expansion = get_augmentations(img, expansion_size, 'perspective', get_img=True)
        embedding = model(expansion, get_feature=True)
        embeddings.append(embedding.detach().cpu())
        labels.append(996)

        expansion = get_augmentations(img, expansion_size, 'crop+rotate', get_img=True)
        embedding = model(expansion, get_feature=True)
        embeddings.append(embedding.detach().cpu())
        labels.append(997)

        expansion = trainer.model.pick_animals(img, expansion_size=expansion_size,\
                                                random=args.random_picker, get_original=False, type = 'mix-up', get_img=True)
        embedding = model(expansion, get_feature=True)
        embeddings.append(embedding.detach().cpu())
        labels.append(998)

        expansion = trainer.model.pick_animals(img, expansion_size=expansion_size,\
                                                random=args.random_picker, get_original=False, type = 'funit', get_img=True)
        embedding = model(expansion, get_feature=True)
        embeddings.append(embedding.detach().cpu())
        labels.append(999)
embeddings = np.concatenate(embeddings)
labels = np.concatenate(labels).reshape(-1, 1)
# print(labels.shape)
with open('./analysis/embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
with open('./analysis/embeddings_label.pkl', 'wb') as f:
    pickle.dump(labels, f)