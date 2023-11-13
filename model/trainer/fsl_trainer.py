import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.FUNIT.utils import get_train_loaders, get_config, get_dichomy_loaders, loader_from_list
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval, get_augmentations
)
# from tensorboardX import SummaryWriter
from collections import deque
from tqdm import tqdm

class FSLTrainer(Trainer):
    def __init__(self, args, config):
        super().__init__(args)
        loaders = get_dichomy_loaders(config)
        self.config = config

        self.test_loader_funit = loaders[1]
        self.test_loader_fsl = loaders[2]

        conf = get_config('/home/yunwei/new/FSL-Rectifier/picker.yaml') ## slack shift
        self.train_loader_funit = loader_from_list(
            root=conf['data_folder_train'],
            file_list=conf['data_list_train'],
            batch_size=conf['pool_size'],
            new_size=140,
            height=128,
            width=128,
            crop=True,
            num_workers=4,
            dataset=conf['dataset'])
        # self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux
    
    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        
        # start FSL training
        label, label_aux = self.prepare_label()
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        
        tl1 = Averager()
        tl2 = Averager()
        ta = Averager()

        for it, batch in enumerate(self.train_loader):
        # for batch in self.train_loader:
            self.train_step += 1

            if torch.cuda.is_available():
                data, gt_label = [_.cuda() for _ in batch]
            else:
                data, gt_label = batch[0], batch[1]
            
            data_tm = time.time()

            # get saved centers
            logits, reg_logits = self.para_model(data)
            if reg_logits is not None:
                loss = F.cross_entropy(logits, label)
                total_loss = loss + args.balance * F.cross_entropy(reg_logits, label_aux)
            else:
                loss = F.cross_entropy(logits, label)
                total_loss = F.cross_entropy(logits, label)
                
            tl2.add(loss)
            forward_tm = time.time()
            self.ft.add(forward_tm - data_tm)
            acc = count_acc(logits, label)

            tl1.add(total_loss.item())
            ta.add(acc)

            self.optimizer.zero_grad()
            total_loss.backward()
            backward_tm = time.time()
            self.bt.add(backward_tm - forward_tm)

            self.optimizer.step()
            optimizer_tm = time.time()
            self.ot.add(optimizer_tm - backward_tm)    

            # refresh start_tm
            # self.lr_scheduler.step()
            # self.try_evaluate(epoch)

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args
        # config = get_config('/home/nus/Documents/research/augment/code/FEAT/model/FUNIT/configs/picker.yaml') ## slack shift to train_fsl
        # config['batch_size'] = args.query + 1
        from model.FUNIT.trainer import FUNIT_Trainer
        trainer = FUNIT_Trainer(self.config)
        trainer.cuda()
        trainer.load_ckpt('animal_pretrained.pt')
        trainer.eval()
        self.trainer = trainer
        picker = FUNIT_Trainer(self.config)
        picker.cuda()
        picker.load_ckpt('/home/yunwei/new/FSL-Rectifier/outputs/picker/gen_10999.pt')
        picker.eval()
        picker = picker.model.gen
        self.picker = picker
        # evaluation mode
        # self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        # path = '/home/yunwei/new/FSL-Rectifier/Animals-ConvNet-Pre/0.01_0.1_[75, 150, 300]/checkpoint.pth'
        # path = '/home/yunwei/new/FSL-Rectifier/Animals-Res12-Pre/0.1_0.1_[75, 150, 300]/checkpoint.pth'
        loaded_dict = torch.load(args.model_path)['state_dict']
        new_params = {}
        keys = list(loaded_dict.keys())
        for key_i, k in enumerate(self.model.state_dict().keys()):
            # if 'encoder' in k:
            #     k = 'encoder.layer' + '.'.join(k.split('.')[3:])
            new_params[k] = loaded_dict[keys[key_i]]
        assert key_i == len(keys) - 3
        self.model.load_state_dict(new_params) ## arg path
        self.model.eval()
        baseline = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        oracle = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        # i2name = {0: 'mix-up', 1: 'funit', 2: 'color', 3:'affine' , 4: 'crop+rotate', 5: 'random-mix-up'}
        # i2name = {0: 'mix-up', 1: 'funit', 2: 'color', 3:'affine' , 4: 'crop+rotate'}
        # i2name = {0: 'random-funit', 1: 'funit', 2: 'affine'}
        # i2name = {0: 'mix-up', 1: 'random-funit', 2: 'color', 3:'affine' , 4: 'crop+rotate', 5: 'funit'}
        i2name = {0: 'color', 1:'affine' , 2: 'crop+rotate', 3: 'funit'}
        expansion_res = []
        for i in i2name:
            expansion_res.append(np.zeros((args.num_eval_episodes, 2))) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        # print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
        #         self.trlog['max_acc_epoch'],
        #         self.trlog['max_acc'],
        #         self.trlog['max_acc_interval']))
        qry_expansion = args.qry_expansion
        spt_expansion = args.spt_expansion
        old_shot = args.eval_shot
        old_qry = args.eval_query
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader_fsl):
            # for i, batch in tqdm(enumerate(self.test_loader, 1)):
                if i >= args.num_eval_episodes:
                    break
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                new_data = torch.empty(data.shape).cuda()
                # print(new_data.shape)
                
                # get baseline
                # for img_i in range(len(new_data)):
                #     img = data[img_i].unsqueeze(0)
                #     new_data[img_i] = trainer.model.pick_animals(picker, img,\
                #                  self.train_loader_funit, expansion_size=0, random=args.random_picker)
                # logits = self.model(new_data)
                # loss = F.cross_entropy(logits, label)
                # acc = count_acc(logits, label)
                # baseline[i-1, 0] = loss.item()
                # baseline[i-1, 1] = acc

                # oracle_new_data = torch.empty(oracle_data.shape).cuda()
                # for img_i in range(spt_expansion * 5):
                #     img = oracle_data[img_i].unsqueeze(0)
                #     oracle_new_data[img_i] = trainer.model.pick_animals(picker, img,\
                #                  self.train_loader_funit, expansion_size=0, random=args.random_picker)
                # oracle_new_data[spt_expansion * 5:] = new_data
                # logits = self.model(oracle_new_data)
                # loss = F.cross_entropy(logits, label)
                
                # acc = count_acc(logits, label)
                new_data = data
                oracle[i-1, 0] = 0#loss.item()
                oracle[i-1, 1] = 0#acc
                
                # old data shape: 80, 3, 128, 128
                # old_spt = torch.empty(5 * shot, data.shape[1], data.shape[2], data.shape[3]).\
                #     cuda()
                for type_i in i2name:
                    name = i2name[type_i]
                    # print(f'getting results for {name}')
                    # expand support 
                    original_spt = data[:5, :, :, :]
                    reconstructed_spt = new_data[:5, :, :, :]
                    # expand queries
                    
                    
                    if spt_expansion == 0:
                        new_spt = reconstructed_spt
                    elif name in ['funit', 'mix-up', 'random-mix-up', 'random-funit']: # use original data
                        new_spt = self.get_class_expansion(picker, original_spt, spt_expansion, type=name)                   
                    else:
                        new_spt = self.get_class_expansion(picker, reconstructed_spt, spt_expansion, type=name)

                    original_qry = new_data[5:, :, :, :]
                    new_qries = torch.empty(old_qry, 5 * qry_expansion, data.shape[1], data.shape[2], data.shape[3]).\
                        cuda()
                    k = 0
                    if name in ['funit', 'mix-up', 'random-mix-up', 'random-funit']: # use original data
                        new_spt = self.get_class_expansion(picker, original_spt, spt_expansion, type=name)
                        for class_chunk_i in range(5, len(data), 5):
                            class_chunk = data[class_chunk_i:class_chunk_i+5]
                            new_qries[k] = self.get_class_expansion(picker, class_chunk, qry_expansion, type=name)
                            k += 1
                    else:# use restructured data
                        new_spt = self.get_class_expansion(picker, reconstructed_spt, spt_expansion, type=name)
                        for class_chunk_i in range(5, len(new_data), 5):
                            class_chunk = new_data[class_chunk_i:class_chunk_i+5]
                            new_qries[k] = self.get_class_expansion(picker, class_chunk, qry_expansion, type=name)
                            k += 1
                    assert k == old_qry
                    new_qries = new_qries.flatten(end_dim=1)

                    expanded_data = torch.cat([original_spt, new_spt, original_qry, new_qries], dim=0)
                    logits = self.model(expanded_data, qry_expansion=qry_expansion, spt_expansion=spt_expansion)
                    loss = F.cross_entropy(logits, label)
                    acc = count_acc(logits, label)
                    expansion_res[type_i][i-1, 0] = loss.item()
                    expansion_res[type_i][i-1, 1] = acc

        assert(i == baseline.shape[0])
        vl, _ = compute_confidence_interval(baseline[:,0])
        va, vap = compute_confidence_interval(baseline[:,1])
        
        # print()
        result_str = ''
        result_str += 'Baseline Test acc={:.4f} + {:.4f}\n'.format(va, vap)

        vl, _ = compute_confidence_interval(oracle[:,0])
        va, vap = compute_confidence_interval(oracle[:,1])
        
        result_str += 'Oracle Test acc={:.4f} + {:.4f}\n'.format(va, vap)
        # print()
        
        for type_i in i2name:
            name = i2name[type_i]
            vl, _ = compute_confidence_interval(expansion_res[type_i][:,0])
            va, vap = compute_confidence_interval(expansion_res[type_i][:,1])
            
            result_str += f'{name} Test acc={va} + {vap}\n'
            # print(f'{name} Test acc={va} + {vap}\n')

        with open(f'./outputs/{args.model_class}-{args.backbone_class}-{args.dataset}-{args.use_euclidean}-record.txt', 'w') as file:
            file.write(result_str)
        return vl, va, vap
    
    # input 01234, return 012340123401234
    # 0: 'oracle', 1: 'mix_up', 2: 'affine', 3: 'color', 4: 'crops_flip_scale', 5: 'funit'
    def get_class_expansion(self, picker, data, expansion, type='funit'):
        expanded = torch.empty(5, expansion, data.shape[1], data.shape[2], data.shape[3]).cuda()
        for class_i in range(5):
            img = data[class_i].unsqueeze(0)
            if type == 'funit' or type == 'mix-up':
                class_expansions = self.trainer.model.pick_animals(self.picker, img, self.train_loader_funit, \
                        expansion_size=expansion, random=False, get_img=False, get_original=False, type=type)
            elif type == 'random-mix-up' or type == 'random-funit':
                class_expansions = self.trainer.model.pick_animals(self.picker, img, self.train_loader_funit, \
                        expansion_size=expansion, random=True, get_img=False, get_original=False, type=type)
            else:
                class_expansions = get_augmentations(img, expansion, type, get_img=False)
            expanded[class_i] = class_expansions
        expanded = expanded.swapaxes(0, 1).flatten(end_dim=1)
        # expanded = expanded.reshape(5 * expansion, data.shape[1], data.shape[2], data.shape[3])
        return expanded
        
    def final_record(self):
        # save the best performance in a txt file
        
        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))            