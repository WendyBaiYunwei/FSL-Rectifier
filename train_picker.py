import torch
import os
import sys
import argparse
import shutil

from model.FUNIT.utils import get_config, get_train_loaders, make_result_folders, reorganize_data
from model.FUNIT.utils import write_loss, write_html, write_1images, Timer, get_dichomy_loaders
from model.FUNIT.trainer import FUNIT_Trainer

import torch.backends.cudnn as cudnn
# Enable auto-tuner to find the best algorithm to use for your hardware.
cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='/home/nus/Documents/research/augment/code/FEAT/model/FUNIT/configs/picker.yaml',
                    help='configuration file for training and testing')
parser.add_argument('--output_path',
                    type=str,
                    default='.',
                    help="outputs path")
parser.add_argument('--multigpus',
                    action="store_true")
parser.add_argument('--batch_size',
                    type=int,
                    default=0)
parser.add_argument('--test_batch_size',
                    type=int,
                    default=4)
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
# Override the batch size if specified.
if opts.batch_size != 0:
    config['batch_size'] = opts.batch_size

trainer = FUNIT_Trainer(config)
trainer.cuda()
if opts.multigpus:
    ngpus = torch.cuda.device_count()
    config['gpus'] = ngpus
    print("Number of GPUs: %d" % ngpus)
    trainer.model = torch.nn.DataParallel(
        trainer.model, device_ids=range(ngpus))
else:
    config['gpus'] = 1

loaders = get_dichomy_loaders(config)
train_loader = loaders[0]

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
# train_writer = SummaryWriter(
#     os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = make_result_folders(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

iterations = trainer.resume(checkpoint_directory,
                            hp=config,
                            multigpus=opts.multigpus) # compulsory resume

while True:
    for it, data in enumerate(train_loader):
        # data (batch, way, 3, 128, 128)
        co_data, cl_data = reorganize_data(data)
        with Timer("Elapsed time in update: %f"):
            loss = trainer.picker_update(co_data, cl_data, config)
            # g_acc = trainer.gen_update(co_data, cl_data, cn_data, config,
            #                            opts.multigpus)
            torch.cuda.synchronize()
            print('loss: %.4f' % (loss))

        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            # write_loss(iterations, trainer, train_writer)

        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            print('change checkpoint dir', checkpoint_directory)
            exit()
            trainer.save(checkpoint_directory, iterations, opts.multigpus)
            print('Saved model at iteration %d' % (iterations + 1))

        iterations += 1
        if iterations >= max_iter:
            print("Finish Training")
            sys.exit(0)
