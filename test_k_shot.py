"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from model.FUNIT.utils import get_config
from model.FUNIT.trainer import FUNIT_Trainer

import argparse

from skimage import exposure, io

def default_loader(path):
    # image = Image.open(path).convert('RGB')
    # return image ##slack
    image = io.imread(path)
    image = exposure.equalize_adapthist(image, clip_limit=0.1)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    return image

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='/home/nus/Documents/research/augment/code/FEAT/model/FUNIT/configs/funit_traffic_signs.yaml')
parser.add_argument('--ckpt',
                    type=str,
                    default="/home/nus/Documents/research/augment/code/FEAT/outputs/funit_traffic_signs/checkpoints/gen_99999.pt")
                    # default='pretrained/animal119_gen_00100000.pt')
parser.add_argument('--class_image_folder',
                    type=str,
                    default='/home/nus/Documents/research/augment/code/FEAT/model/FUNIT/images/traffics/class_styles')
parser.add_argument('--input',
                    type=str,
                    default='/home/nus/Documents/research/augment/code/FEAT/model/FUNIT/images/traffics/input_content.jpg')
                    # default='images/input_content.jpg')
parser.add_argument('--output',
                    type=str,
                    default='/home/nus/Documents/research/augment/code/FEAT/model/FUNIT/images/traffics/output.jpg')
                    # default='images/output.jpg')
opts = parser.parse_args()
cudnn.benchmark = True
opts.vis = True
config = get_config(opts.config)
config['batch_size'] = 1
config['gpus'] = 1

trainer = FUNIT_Trainer(config)
trainer.cuda()
trainer.load_ckpt(opts.ckpt)
trainer.eval()

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_list = [transforms.Resize((128, 128))] + transform_list
transform = transforms.Compose(transform_list)

print('Compute average class codes for images in %s' % opts.class_image_folder)
images = os.listdir(opts.class_image_folder)
for i, f in enumerate(images):
    fn = os.path.join(opts.class_image_folder, f)
    # img = Image.open(fn).convert('RGB')
    img = default_loader(fn)
    img_tensor = transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        class_code = trainer.model.compute_k_style(img_tensor, 1)
        if i == 0:
            new_class_code = class_code
        else:
            new_class_code += class_code
final_class_code = new_class_code / len(images)
# image = Image.open(opts.input)
# image = image.convert('RGB')
image = default_loader(opts.input)
image.save('/home/nus/Documents/research/augment/code/FEAT/model/FUNIT/images/traffics/initial_out.jpg', 'JPEG', quality=99)
content_img = transform(image).unsqueeze(0)

print('Compute translation for %s' % opts.input)
with torch.no_grad():
    output_image = trainer.model.translate_simple(content_img, final_class_code)
    image = output_image.detach().cpu().squeeze().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = ((image + 1) * 0.5 * 255.0)
    output_img = Image.fromarray(np.uint8(image))
    output_img.save(opts.output, 'JPEG', quality=99)
    print('Save output to %s' % opts.output)
