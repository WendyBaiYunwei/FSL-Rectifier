# creates and saves the recon/translation images of the test imgs (3 copies)
# saved format:
# path_prefix/n02106030/n02106030_16184.JPEG_109_179_237_312_trans{1,2,3}.jpg
# path_prefix/n02106030/n02106030_16184.JPEG_109_179_237_312_recon.jpg

import numpy as np
from PIL import Image
import torch
from model.image_translator.utils import loader_from_list, get_config
from model.image_translator.trainer import Translator_Trainer
from model.utils import pick_translate

expansion_size = 3

config = get_config('./animals.yaml') 
config['batch_size'] = 1

image_translator = Translator_Trainer(config)
image_translator.load_ckpt('animals_gen.pt')
image_translator = image_translator.model.cuda()
image_translator.eval()

picker = Translator_Trainer(config).cuda()
picker.load_ckpt('animals_picker.pt')
picker_loader = loader_from_list(
    root=config['data_folder_train'],
    file_list=config['data_list_train'],
    batch_size=config['pool_size'],
    new_size=140,
    height=128,
    width=128,
    crop=True,
    num_workers=4,
    dataset=config['dataset'])
picker = picker.model.gen
picker.eval()

testloader = loader_from_list(
    root=config['data_folder_test'],
    file_list=config['data_list_test'],
    batch_size=config['batch_size'],
    new_size=140,
    height=128,
    width=128,
    crop=True,
    num_workers=4,
    return_paths=True,
    dataset=config['dataset']) 

for i, data in enumerate(testloader):
    if i % 10 == 0:
        print(f'{i} / {len(testloader)}')
    original_img = data[0].cuda(1).squeeze()
    label = data[1]
    paths = data[2]

    imgs = pick_translate(image_translator, picker, original_img, picker_loader,\
            expansion_size=expansion_size, get_original=True)
    # save image
    for selected_i in range(expansion_size):
        translation = imgs[selected_i]
        image = translation.detach().cpu().squeeze().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = ((image + 1) * 0.5 * 255.0)
        output_img = Image.fromarray(np.uint8(image))
        
        original_path = '.'.join(paths[0].split('.')[:-1])
        if selected_i == 0:
            output_img.save(\
            f'{original_path}_recon.jpg', 'JPEG', quality=99)
        else:
            output_img.save(\
            f'{original_path}_trans{selected_i}.jpg', 'JPEG', quality=99)
