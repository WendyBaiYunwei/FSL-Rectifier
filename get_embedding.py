import pickle
import numpy as np
from model.FUNIT.utils import loader_from_list, get_config
from model.utils import get_augmentations
from model.trainer.helpers import (
    prepare_model
)
from model.FUNIT.trainer import FUNIT_Trainer
import argparse
import torch
from model.models.feat import FEAT

# get 100 randomly sampled animals embeddings
sample_iters = 1000#200 * 10
# path = '/home/yunwei/new/FSL-Rectifier/Animals-ConvNet-Pre/0.01_0.1_[75, 150, 300]/checkpoint.pth'
path = '/home/yunwei/new/FSL-Rectifier/checkpoints/Animals-FEAT-ConvNet-05w01s15q-DIS/20_0.5_lr0.01mul10_step_T11T21_b0_bsz080-NoAug/epoch-last.pth'
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
parser.add_argument('--init_weights', type=str, default=path)
parser.add_argument('--multi_gpu', type=str, default=None)
args = parser.parse_args()

# get loader
# get the embeddings in numpy
# store the embeddings
config = get_config('./picker.yaml')
loader = loader_from_list(
    root=config['data_folder_test'],
    file_list=config['data_list_test'],
    batch_size=1,
    new_size=140,
    height=128,
    width=128,
    crop=True,
    num_workers=1,
    dataset=args.dataset)

train_loader = loader_from_list(
    root=config['data_folder_train'],
    file_list=config['data_list_train'],
    batch_size=config['pool_size'],
    new_size=140,
    height=128,
    width=128,
    crop=True,
    num_workers=4,
    dataset=config['dataset'])
args.num_class = 119
model = FEAT(args)
loaded_dict = torch.load(path)['params']
new_params = {}
keys = list(loaded_dict.keys())
# print(keys)
# print(self.model.state_dict().keys())
# exit()
for key in model.state_dict().keys():
    # if 'encoder' in k:
    #     k = 'encoder.layer' + '.'.join(k.split('.')[3:])
    if key in keys:
        new_params[key] = loaded_dict[key]
    else:
        new_params[key] = model.state_dict()[key]
# assert key_i == len(keys) - 3
model.load_state_dict(new_params) ## arg path

trainer = FUNIT_Trainer(config)
trainer.cuda()
trainer.load_ckpt('/home/yunwei/new/FSL-Rectifier/animal_pretrained.pt')
trainer.eval()

picker = FUNIT_Trainer(config)
picker.cuda()
picker.load_ckpt('/home/yunwei/new/FSL-Rectifier/outputs/picker/gen_10999.pt')
picker = picker.model.gen
picker.eval()

model = model.cuda()
embeddings = []
labels = []
expansion_size = 5
need_expansion = True
for i, data in enumerate(loader):
    if i%100 == 0:
        print(i)
    img = data[0].cuda()
    label = data[1].detach().cpu()
    keep_idx = label < 10
    # print(label.shape)
    # exit()
    img = img[keep_idx]
    if len(img) == 0:
        continue
    label = label[keep_idx]
    # print(label)
    # exit()
    if i >= sample_iters: ## sample_iters
        break
    reconstructed_img = trainer.model.pick_animals(picker, img, train_loader, expansion_size=0)
    embedding = model(reconstructed_img.unsqueeze(0), get_feature=True)
    embeddings.append(embedding.detach().cpu())
    # print(label)
    labels.append(label)
    if label == 0 and need_expansion == True: # get expansion
        labels.pop()
        label = torch.full(label.shape, 995) # for the random point
        labels.append(label.detach().cpu())
        need_expansion = False

        # expansion = get_augmentations(reconstructed_img.unsqueeze(0), expansion_size*2, 'color')
        # embedding = model(expansion, get_feature=True)
        # embeddings.append(embedding.detach().cpu())
        # label = torch.full((expansion_size*2,), 996)
        # labels.append(label.detach().cpu())
        # # labels.extend([label.detach().cpu() for _ in range(expansion_size)])

        crop_expansion1 = get_augmentations(reconstructed_img.unsqueeze(0), expansion_size, 'crop+rotate')
        # crop_expansion2 = get_augmentations(reconstructed_img.unsqueeze(0), expansion_size, 'crop+rotate')
        # embedding = model(torch.cat([crop_expansion1, crop_expansion2], dim=0), get_feature=True)
        # embeddings.append(embedding.detach().cpu())
        # label = torch.full((expansion_size*2,), 997)
        # labels.append(label.detach().cpu())
        # labels.extend([label.detach().cpu() for _ in range(expansion_size)])

        # expansion = trainer.model.pick_animals(picker, img, train_loader, expansion_size=expansion_size*2,\
        #                                         random=False, get_original=False, type = 'mix-up')
        # embedding = model(expansion, get_feature=True)
        # embeddings.append(embedding.detach().cpu())
        # label = torch.full((expansion_size*2,), 998)
        # labels.append(label.detach().cpu())
        # labels.extend([label.detach().cpu() for _ in range(expansion_size)])

        expansion = trainer.model.pick_animals(picker, img, train_loader, expansion_size=expansion_size,\
                                                random=False, get_original=False, type = 'funit')
        embedding = model(torch.cat([crop_expansion1, expansion], dim=0), get_feature=True)
        embeddings.append(embedding.detach().cpu())
        label = torch.full((expansion_size*2,), 999)
        labels.append(label.detach().cpu())
        # labels.extend([label.detach().cpu() for _ in range(expansion_size)])
embeddings = np.concatenate(embeddings)
print(embeddings.shape)
labels = np.concatenate(labels).reshape(-1, 1)
print(labels.shape)
# print(labels.shape)
with open('./analysis/embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
with open('./analysis/embeddings_label.pkl', 'wb') as f:
    pickle.dump(labels, f)