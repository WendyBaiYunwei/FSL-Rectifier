import os
import shutil
import time
import pprint
import torch
import argparse
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import cv2
from model.image_translator.utils import get_recon, get_orig

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()    
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)

    return encoded_indicies

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def ensure_path(dir_path, scripts_to_save=None):
    if os.path.exists(dir_path):
        if input('{} exists, remove? ([y]/n)'.format(dir_path)) != 'n':
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
    else:
        os.mkdir(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for src_file in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(src_file))
            print('copy {} to {}'.format(src_file, dst_file))
            if os.path.isdir(src_file):
                shutil.copytree(src_file, dst_file)
            else:
                shutil.copyfile(src_file, dst_file)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def get_sim_scores_model(qry, nbs, model):
    with torch.no_grad():
        features = model(torch.cat([qry, nbs]), get_feature=True)
    qry = features[0]
    nbs = features[1:]
    similarities = []
    for nb in nbs:
        similarity = F.cosine_similarity(qry, nb, dim=0)
        similarities.append(similarity.item())
    return torch.tensor(similarities).cuda()

def get_sim_scores_orb(qry, nbs):
    query_image = cv2.imread(qry, cv2.IMREAD_GRAYSCALE)
    new_height = 84
    new_width = 84
    query_image = cv2.resize(query_image, (new_width, new_height))

    # Initialize feature extractor
    orb = cv2.SIFT_create()

    # Extract keypoints and descriptors from the query image
    keypoints_query, descriptors_query = orb.detectAndCompute(query_image, None)

    # Initialize a brute-force matcher
    bf = cv2.BFMatcher()

    scores = []
    
    
    # Iterate through dataset images
    for image_path in nbs:
        # Load dataset image
        dataset_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        dataset_image = cv2.resize(dataset_image, (new_width, new_height))
        # exit()
        
        # Extract keypoints and descriptors from the dataset image
        keypoints_dataset, descriptors_dataset = orb.detectAndCompute(dataset_image, None)

        if descriptors_dataset is None or descriptors_query is None:
            print('SIFT failed for one pair of comparison')
            score = 0
        else:
            min_length = min(len(descriptors_dataset), len(descriptors_query))
            descriptors_query_clipped = descriptors_query[:min_length]
            descriptors_dataset = descriptors_dataset[:min_length]
            # Match descriptors between query and dataset images
            matches = bf.match(descriptors_query_clipped, descriptors_dataset)
        
            # Calculate the similarity score (number of matches)
            score = len(matches)
        scores.append(score)

    return torch.tensor(scores).cuda()

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def postprocess_args(args):            
    args.num_classes = args.way
    save_path1 = '-'.join([args.dataset, args.model_class, args.backbone_class, '{:02d}w{:02d}s{:02}q'.format(args.way, args.shot, args.query)])
    save_path2 = '_'.join([str('_'.join(args.step_size.split(','))), str(args.gamma),
                           'lr{:.2g}mul{:.2g}'.format(args.lr, args.lr_mul),
                           str(args.lr_scheduler), 
                           'T1{}T2{}'.format(args.temperature, args.temperature2),
                           'b{}'.format(args.balance),
                           'bsz{:03d}'.format( max(args.way, args.num_classes)*(args.shot+args.query) ),
                           # str(time.strftime('%Y%m%d_%H%M%S'))
                           ])    
    if args.init_weights is not None:
        save_path1 += '-Pre'
    if args.use_euclidean:
        save_path1 += '-DIS'
    else:
        save_path1 += '-SIM'
            
    if args.fix_BN:
        save_path2 += '-FBN'
    if not args.augment:
        save_path2 += '-NoAug'
            
    if not os.path.exists(os.path.join(args.save_dir, save_path1)):
        os.mkdir(os.path.join(args.save_dir, save_path1))
    args.save_path = os.path.join(args.save_dir, save_path1, save_path2)
    return args

def get_command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=10000)
    parser.add_argument('--model_class', type=str, default='FEAT', 
                        choices=['MatchNet', 'ProtoNet', 'BILSTM', 'DeepSet', 'GCN', 'FEAT', 'FEATSTAR', 'SemiFEAT', 'SemiProtoFEAT']) # None for MatchNet or ProtoNet
    parser.add_argument('--use_euclidean', action='store_true', default=False)    
    parser.add_argument('--backbone_class', type=str, default='ConvNet',
                        choices=['ConvNet', 'Res12', 'Res18', 'WRN'])
    parser.add_argument('--dataset', type=str, default='animals',
                        choices=['cub', 'animals', 'animals-buffer', 'cub-buffer', 'miniImagenet', 'miniImagenet-buffer'])
    
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--eval_shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=1)
    parser.add_argument('--eval_query', type=int, default=1)
    parser.add_argument('--balance', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--temperature2', type=float, default=1)  # the temperature in the  
     
    # optimization parameters
    parser.add_argument('--orig_imsize', type=int, default=-1) # -1 for no cache, and -2 for no resize, only for cub and CUB
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_mul', type=float, default=10)    
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.2)    
    parser.add_argument('--fix_BN', action='store_true', default=False)     # means we do not update the running mean/var in BN, not to freeze BN
    parser.add_argument('--augment',   action='store_true', default=False)
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--init_weights', type=str, default=None)
    
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--random_picker', action='store_true', default=True)
    parser.add_argument('--qry_expansion', type=int, default=0)
    parser.add_argument('--spt_expansion', type=int, default=0)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--add_transform', type=str, choices=['perspective', 'crop+rotate', 'original'], default=None)
    parser.add_argument('--aug_type', type=str, choices=['random-image_translator', 'image_translator', 'mix-up', \
        'random-mix-up', 'sim-mix-up', 'crop+rotate', 'original', 'true-test', 'color', 'affine'], default=None)
    parser.add_argument('--note', type=str, default='')
    
    return parser

# input: one img 3x84x84, output: augmentations
def get_augmentations(img, expansion, type, get_img=False):
    expansions = torch.empty(expansion, img.shape[0], img.shape[1], img.shape[2]).cuda()
    crop_rotate = transforms.Compose([
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomCrop(size=(84, 84))
    ])
    transformations = {
        'affine': transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
        'crop+rotate': crop_rotate,
        'color': transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        'original': transforms.Resize(84)
    }
    for expansion_i in range(expansion):
        augmented_image = transformations[type](img)
        expansions[expansion_i] = augmented_image
        if get_img == True:
            image = F.to_pil_image(augmented_image)
            image.save(f'augmented_image_{expansion_i}.jpg')
    return expansions

def pick_mixup(qry_img, qry, picker_loader, model, expansion_size=0, get_img = False, \
    random=False, get_original=True, augtype='mix-up', picker=None): 
    if expansion_size == 0:
        assert get_original == True
    candidate_neighbours = next(iter(picker_loader)) # from train sampler, size: pool_size, 3, h, w + label_size
    nb_images = candidate_neighbours[0].cuda() # extracts img from img+label+name
    candidate_neighbours = candidate_neighbours[2] # extracts name from img+label+name
    
    assert len(nb_images) >= expansion_size
    if picker is not None:
        with torch.no_grad():
            qry_features = picker.enc_content(qry_img).mean((1,2)).reshape(1, -1) # batch=1, feature_size
            nb_features = picker.enc_content(nb_images).mean((2,3))
            nb_features_trans = nb_features.transpose(1,0)
            scores = torch.mm(qry_features, nb_features_trans).squeeze() # best (lower KL divergence) should be in front after sorting
    else:
        qry_img = qry_img.squeeze()
        scores = get_sim_scores_model(qry_img.unsqueeze(0), nb_images, model)
    if random == False and picker is not None:
        scores, idxs = torch.sort(scores, descending=False)
        idxs = idxs.long()
        selected_nbs = [candidate_neighbours[i.item()] for i in idxs]
        selected_nbs = selected_nbs[:expansion_size]
    elif random == False and picker is None:
        scores, idxs = torch.sort(scores, descending=True)
        idxs = idxs.long()
        selected_nbs = nb_images.index_select(dim=0, index=idxs)
        selected_nbs = selected_nbs[:expansion_size]
    else:
        selected_nbs = nb_images[:expansion_size]
    if picker is not None: # animal dataset
        qry = get_orig(qry)
        if get_original == True:
            translations = [qry]
        else:
            translations = []
        for selected_i in range(expansion_size):
            nb_name = selected_nbs[selected_i]
            nb = get_orig(nb_name)
            translation = 0.5*(nb + qry)
            translations.append(translation)
    else:
        qry = qry_img
        if get_original == True:
            translations = [qry]
        else:
            translations = []
        for selected_i in range(expansion_size):
            nb = selected_nbs[selected_i]
            translation = 0.5*(nb + qry)
            translations.append(translation)
    return torch.stack(translations).squeeze()

def pick_nb(qry_img, qry, picker_loader, model, expansion_size=0, get_img = False, \
    random=False, get_original=True, augtype='mix-up', picker=None): 
    if expansion_size == 0:
        get_original = True
    candidate_neighbours = next(iter(picker_loader)) # from train sampler, size: pool_size, 3, h, w + label_size
    nb_images = candidate_neighbours[0].cuda() # extracts img from img+label+name
    candidate_neighbours = candidate_neighbours[2] # extracts name from img+label+name

    assert len(nb_images) >= expansion_size
    
    qry_img = qry_img.squeeze()
    scores = get_sim_scores_model(qry_img.unsqueeze(0), nb_images, model)
    scores, idxs = torch.sort(scores, descending=True)
    idxs = idxs.long()
    candidate_neighbours = [candidate_neighbours[idx] if \
        scores[idx] > 0.9 else qry for idx in idxs]
    if get_original == True:
        nbs = [qry_img]
    else:
        nbs = []
    for selected_i in range(expansion_size):
        nb = candidate_neighbours[selected_i] ##todo
        nbs.append(nb)
    return nbs
    
        
def pick_translate(translator, picker, qry, picker_loader, expansion_size=0, get_img = False, \
    random=False, get_original=False, augtype='image_translator'): 
    if expansion_size == 0:
        assert get_original == True
    candidate_neighbours = next(iter(picker_loader)) # from train sampler, size: pool_size, 3, h, w + label_size
    candidate_neighbours = candidate_neighbours[0].cuda() # extracts img from img+label
    assert len(candidate_neighbours) >= expansion_size
    with torch.no_grad():
        new_size = (128, 128)
        qry = qry.unsqueeze(0)
        qry = F.interpolate(qry, size=new_size, mode='bilinear', align_corners=False)
        candidate_neighbours = F.interpolate(candidate_neighbours, size=new_size,\
             mode='bilinear', align_corners=False)

        qry_features = picker.enc_content(qry).mean((2,3)) # batch=1, feature_size
        nb_features = picker.enc_content(candidate_neighbours).mean((2,3))
        nb_features_trans = nb_features.transpose(1,0)
        scores = torch.mm(qry_features, nb_features_trans).squeeze() # q qries, n neighbors
    if random == False:
        scores, idxs = torch.sort(scores, descending=False) # best (lower KL divergence) in front
        idxs = idxs.long()
        selected_nbs = candidate_neighbours.index_select(dim=0, index=idxs)
        selected_nbs = selected_nbs[:expansion_size, :, :, :]
    else:
        selected_nbs = candidate_neighbours[:expansion_size, :, :, :]
    class_code = translator.compute_k_style(qry, 1)
    qry = translator.translate_simple(qry, class_code)
    if get_original == True:
        translations = [qry]
    else:
        translations = []
    
    for selected_i in range(expansion_size):
        nb = selected_nbs[selected_i, :, :, :].unsqueeze(0)
        if augtype == 'image_translator' or augtype == 'random-image_translator':
            translation = translator.translate_simple(nb, class_code)
            translation = F.interpolate(translation, size=(84,84),\
                mode='bilinear', align_corners=False)
        elif augtype == 'mix-up' or augtype == 'random-mix-up':
            nb = translator.translate_simple(nb, translator.compute_k_style(nb, 1))
            translation = 0.5*(nb + qry)
        translations.append(translation)

    if get_img == True:
        return translations
    else:
        return torch.stack(translations).squeeze()
