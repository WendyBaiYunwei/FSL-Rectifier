import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.FUNIT.utils import get_train_loaders, get_config
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
# from tensorboardX import SummaryWriter
from collections import deque
from tqdm import tqdm

class FSLTrainer(Trainer):
    def __init__(self, args, config):
        super().__init__(args)
        loaders = get_train_loaders(config)
        self.config = config
        self.train_loader = loaders[0]
        self.test_loader = loaders[1]
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
        config = get_config('/home/nus/Documents/research/augment/code/FEAT/model/FUNIT/configs/picker.yaml') ## slack shift to train_fsl
        config['batch_size'] = args.query + 1
        from model.FUNIT.trainer import FUNIT_Trainer
        trainer = FUNIT_Trainer(config)
        trainer.cuda()
        trainer.load_ckpt('/home/nus/Documents/research/augment/code/FEAT/model/FUNIT/pretrained/animal119_gen_00100000.pt')
        trainer.eval()
        # evaluation mode
        # self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        # path = '/home/nus/Documents/research/augment/code/FEAT/Animals-ConvNet-Pre/0.01_0.1_[75, 150, 300]/last.pth'
        path = '/home/nus/Documents/research/augment/code/FEAT/Animals-Res12-Pre/0.1_0.1_[75, 150, 300]/last.pth'
        loaded_dict = torch.load(path)['params']
        new_params = {}
        for k in self.model.state_dict().keys():
            new_params[k] = loaded_dict[k]
        self.model.load_state_dict(new_params) ## arg path
        self.model.eval()
        baseline = np.zeros((10, 2)) # loss and acc
        new = np.zeros((10, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        # print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
        #         self.trlog['max_acc_epoch'],
        #         self.trlog['max_acc'],
        #         self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
            # for i, batch in tqdm(enumerate(self.test_loader, 1)):
                print(i)
                if i >= 10:
                    break
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                new_data = torch.empty(data.shape).cuda()
                # print(new_data.shape)
                # exit()
                for img_i in range(len(new_data)):
                    img = data[img_i].unsqueeze(0)
                    new_data[img_i] = trainer.model.pick_animals(img, expansion_size=0, random=True)
                self.args.eval_shot = 1
                logits = self.model(new_data)
                loss = F.cross_entropy(logits, label)
                
                acc = count_acc(logits, label)
                baseline[i-1, 0] = loss.item()
                baseline[i-1, 1] = acc

                # old data shape: 80, 3, 128, 128
                expansion = 2
                self.args.eval_shot = 1 + expansion
                shot = self.args.eval_shot
                multi_shot_data = torch.empty(5 * shot + 5 * 15, data.shape[1], data.shape[2], data.shape[3]).\
                    cuda() # 5 * 2 + 7 * 5 = 10 + 35 = 45
                k = 0
                temp = []
                for img_i in range(len(new_data)):
                    if img_i < 5:
                        img = data[img_i].unsqueeze(0)
                        temp.append(trainer.model.pick_animals(img, \
                            expansion_size=expansion, random=False, get_img=False, img_id=img_i))
                    if img_i == 4:
                        temp = torch.stack(temp).swapaxes(0, 1).flatten(end_dim=1)
                        multi_shot_data[:(img_i + 1) * shot] = temp
                    elif img_i >= 5:
                        multi_shot_data[5 * shot + k] = new_data[img_i]
                        k += 1
                logits = self.model(multi_shot_data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                new[i-1, 0] = loss.item()
                new[i-1, 1] = acc
        assert(i == baseline.shape[0])
        vl, _ = compute_confidence_interval(baseline[:,0])
        va, vap = compute_confidence_interval(baseline[:,1])
        
        print('Baseline Test acc={:.4f} + {:.4f}\n'.format(va, vap))

        vl, _ = compute_confidence_interval(new[:,0])
        va, vap = compute_confidence_interval(new[:,1])
        
        print('New Test acc={:.4f} + {:.4f}\n'.format(va, vap))

        return vl, va, vap
    
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