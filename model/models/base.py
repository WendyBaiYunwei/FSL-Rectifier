import torch
import torch.nn as nn
import numpy as np

class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
        else:
            raise ValueError('')

    def split_instances(self, data):
        args = self.args
        if self.training:
            return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))
        else:
            return  (torch.Tensor(np.arange(args.eval_way*args.eval_shot)).long().view(1, args.eval_shot, args.eval_way), 
                     torch.Tensor(np.arange(args.eval_way*args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query, args.eval_way))

    def forward(self, x, get_feature=False, qry_expansion=0, spt_expansion=0):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x)           #len, emb

            one_shot_embs = instance_embs[:5]
            if spt_expansion > 0:
                mean_sptexp_embs = instance_embs[5:5 * (1 + spt_expansion)].mean(dim=0).unsqueeze(0)
                new_spt = 0.5 * (one_shot_embs + mean_sptexp_embs)
            else:
                new_spt = one_shot_embs

            qry_embs = instance_embs[5 * (1 + spt_expansion):]
            if qry_expansion > 0:
                one_qry_embs = qry_embs[:-5 * qry_expansion]
                mean_qryexp_embs = qry_embs[-5 * qry_expansion:].mean(dim=0).unsqueeze(0)
                new_qry = 0.5 * (one_qry_embs + mean_qryexp_embs)
            else:
                new_qry = qry_embs
            
            new_embs = torch.cat([new_spt, new_qry]).reshape(5 + len(new_qry), -1)
            support_idx, query_idx = self.split_instances(x)
            if self.training:
                logits, logits_reg = self._forward(new_embs, support_idx, query_idx)
                return logits, logits_reg
            else:
                logits = self._forward(new_embs, support_idx, query_idx)
                return logits

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')