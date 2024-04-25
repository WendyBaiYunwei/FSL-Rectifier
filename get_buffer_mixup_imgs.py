# creates and saves the most similar images of the test imgs (3 copies)
# saved format:
# path_prefix/n02106030/n02106030_16184.JPEG_109_179_237_312_sim{1,2,3}.jpg

import numpy as np
from PIL import Image
import torch
from model.trainer.helpers import prepare_model
from model.utils import (
    get_command_line_parser,
    postprocess_args,
    pick_mixup,
)
from model.image_translator.utils import loader_from_list, get_config

expansion_size = 3

config = get_config('./cub.yaml') # change to traffic.yaml for traffic
config['batch_size'] = 1

parser = get_command_line_parser()
args = postprocess_args(parser.parse_args())

# note: please remove normalization in the dataloader first
train_loader_image_translator = loader_from_list(
    root=config['data_folder_train'],
    file_list=config['data_list_train'],
    batch_size=config['pool_size'],
    new_size=84,
    height=84,
    width=84,
    crop=True,
    num_workers=4,
    return_paths=True,
    dataset=config['dataset'])

testloader = loader_from_list(
    root=config['data_folder_train'],
    file_list=config['data_list_train'],
    batch_size=1,
    new_size=84,
    height=84,
    width=84,
    crop=True,
    num_workers=4,
    return_paths=True,
    dataset=config['dataset']) # pre-processing mode set to `Animals` to prevent CLAHE transformations for test samples

for i, data in enumerate(testloader):
    if i % 10 == 0:
        print(f'{i} / {len(testloader)}')
    original_img = data[0].cuda()
    label = data[1]
    paths = data[2]

    model, _ = prepare_model(args)
    imgs = pick_mixup(original_img, paths, train_loader_image_translator, model = model,\
                    expansion_size=expansion_size, random=False, get_img=False, get_original=False, augtype='sim-mix-up')
        
    # save image
    for selected_i in range(expansion_size):
        translation = imgs[selected_i]
        image = translation.detach().cpu().squeeze().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = image * 255.0
        output_img = Image.fromarray(np.uint8(image))
        
        original_path = '.'.join(paths[0].split('.')[:-1])

        output_img.save(\
        f'{original_path}_sim{selected_i}.jpg', 'JPEG', quality=99)
