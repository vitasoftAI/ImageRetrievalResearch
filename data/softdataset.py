from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import random
import numpy as np
import json
import tqdm

def data_split(data_dir, out_path, policy='prod', split=[0.8, 0.1, 0.1]):
    assert sum(split) == 1, 'sum of split should be 1'
    lst = glob.glob(os.path.join(data_dir, 'real/**/*'), recursive=True)
    lst = [i for i in lst if os.path.isfile(i)]
    random.shuffle(lst)
    rslt = {'train':[], 'val':[]}
    if len(split) == 3:
        rslt['test'] = []
    dic = {}
    for i in lst:
        basepath = i.replace(os.path.join(data_dir, ''), '')
        if policy == 'cat':
            pol = basepath.split('/')[1]
        elif policy == 'prod':
            pol = os.path.dirname(basepath).split('_')[1]
        else:
            raise Exception('policy must be one of [cat, prod]')
        if pol not in dic:
            dic[pol] = []
        dic[pol].append(i)
    for value in dic.values():
        idx = int(len(value)*split[1])
        idx = max(idx, 1)
        rslt['val']+= value[:idx]
        if len(split) == 3:
            prev_idx = idx
            idx = int(len(value)*split[2])
            idx = max(idx, 1) + prev_idx
            rslt['test']+= value[prev_idx:idx]
        rslt['train']+= value[idx:]
    with open(out_path, 'w') as f:
        json.dump(rslt, f)
    return out_path

class TripletDataset(Dataset):
    def __init__(self, data_dir, random=True, pos_policy='prod', neg_policy='except_cat', trainval_json=None, trainval=None, data_json=None):
        self.pos_policy = pos_policy
        self.neg_policy = neg_policy
        self.random = random
        self.data_dir = data_dir
        self.cat_idx = {}
        self.prod_idx = {}
        self.pos_neg_dic = {}
        self.neg_dic = {}
        
        if not self.random:
            assert data_json != None, 'data_json is required if not random'
            assert trainval_json == None and trainval == None, 'random false mode doesn\t support trainval mode'
            with open(data_json, 'r') as f:
                json_data = json.loads(f.read())
            self.cat_idx = json_data['meta']['cat_idx']
            self.prod_idx = json_data['meta']['prod_idx']
            self.sketch_lst = json_data['meta']['sketch_lst']
            self.image_lst = json_data['meta']['image_lst']
            self.data = json_data['data']
        else:
            if trainval_json:
                assert trainval != None, 'you should declare whether this is train or val dataset'
                with open(trainval_json, 'r') as f:
                    trainval_data = json.loads(f.read())
                self.image_lst = trainval_data[trainval]
            else:
                self.image_lst = glob.glob(os.path.join(self.data_dir, 'real/**/*'), recursive=True)
            self.sketch_lst = glob.glob(os.path.join(self.data_dir, 'sketch/**/*'), recursive=True)
            self.image_lst = [i for i in self.image_lst if os.path.isfile(i)]
            self.sketch_lst = [i for i in self.sketch_lst if os.path.isfile(i)]

            self.cat_dic, self.prod_dic, self.sketch_dic = {}, {}, {}
            for i in self.sketch_lst+self.image_lst:
                basepath = self.get_basepath(i)
                cat, sketch_name, prod = self.classify(basepath)
                self.cat_dic = self.gen_dic(self.cat_dic, i, cat)
                self.prod_dic = self.gen_dic(self.prod_dic, i, prod)
                self.sketch_dic = self.gen_dic(self.sketch_dic, i, sketch_name)
            for idx, key in enumerate(self.cat_dic.keys()):
                self.cat_idx[key] = idx
            for idx, key in enumerate(self.prod_dic.keys()):
                self.prod_idx[key] = idx
            for qry in self.image_lst:
                basepath = self.get_basepath(qry)
                cat, _, prod = self.classify(basepath)
                if self.pos_policy == 'cat':
                    pos_lst = self.cat_dic[cat]['sketch']
                    pos_policy = cat
                elif self.pos_policy == 'prod':
                    pos_lst = self.prod_dic[prod]['sketch']
                    pos_policy = prod
                else:
                    raise Exception('positive policy must be one of [cat, prod]')
                if self.neg_policy == 'except_cat':
                    neg_policy = cat
                    if neg_policy in self.neg_dic:
                        neg_lst = self.neg_dic[neg_policy]
                    else:
                        neg_lst = list(set(self.sketch_lst) - set(self.cat_dic[neg_policy]['sketch']))
                        self.neg_dic[neg_policy] = neg_lst
                elif self.neg_policy == 'except_prod':
                    neg_policy = prod
                    if neg_policy in self.neg_dic:
                        neg_lst = self.neg_dic[neg_policy]
                    else:
                        neg_lst = list(set(self.sketch_lst) - set(self.prod_dic[neg_policy]['sketch']))
                        self.neg_dic[neg_policy] = neg_lst

                elif self.neg_policy == 'in_cat_except_prod':
                    neg_policy = f'{cat}/{prod}'
                    if neg_policy in self.neg_dic:
                        neg_lst = self.neg_dic[neg_policy]
                    else:
                        pos_prod_lst = self.prod_dic[prod]['sketch']
                        pos_cat_lst = self.cat_dic[cat]['sketch']
                        neg_lst = list(set(pos_cat_lst) - set(pos_prod_lst))
                        self.neg_dic[neg_policy] = neg_lst
                else:
                    raise Exception('negative policy must be one of [except_cat, except_prod, in_cat_except_prod]')
                if pos_lst and neg_lst:
                    self.pos_neg_dic[qry] = {'pos':pos_lst, 'neg': neg_lst, 'pos_policy': pos_policy, 'neg_policy': neg_policy}
            self.image_lst = list(self.pos_neg_dic.keys())
    def __getitem__(self, idx):
        if not self.random:
            return self.data[idx]
        qry = self.image_lst[idx]
        pos_neg = self.pos_neg_dic[self.image_lst[idx]]
        pos_lst, neg_lst, pos_policy, neg_policy = pos_neg['pos'], pos_neg['neg'], pos_neg['pos_policy'], pos_neg['neg_policy']
        
        return {'qry':qry, 'pos':pos_neg['pos'], 'neg':pos_neg['neg'], 'pos_policy': pos_neg['pos_policy'], 'neg_policy': pos_neg['neg_policy']}
    def __len__(self):
        if not self.random:
            return len(self.data)
        return len(self.image_lst)
    def get_basepath(self, path):
        return path.replace(os.path.join(self.data_dir, ''), '')
    def classify(self, path):
        split = path.split('/')
        cat, sketch_name = split[1], os.path.splitext(split[2])[0]
        prod = sketch_name.split('_')[1]
        return cat, sketch_name, prod
    def get_cat_length(self):
        return len(self.cat_dic)
    def get_prod_length(self):
        return len(self.prod_dic)
    
    def gen_dic(self, dic, f_name, parser):
        basepath = self.get_basepath(f_name)
        if parser not in dic:
            dic[parser] = {'sketch': [], 'real': []}
        dic[parser][basepath.split('/')[0]].append(f_name)
        return dic

class TripletImageDataset(TripletDataset):
    def __init__(self, transform_dic=None, pos_return_num=1, neg_return_num=1, load_images=False, **kwargs):
        super(TripletImageDataset, self).__init__(**kwargs)
        self.load_images = load_images
        if self.load_images:
            self.sketch_lst_im = {i: Image.open(i).convert('RGB') for i in self.sketch_lst}
            self.image_lst_im = {i: Image.open(i).convert('RGB') for i in self.image_lst}
        
        self.transform_dic = transform_dic
        if transform_dic:
            self.qry_trans, self.pos_trans, self.neg_trans = transform_dic['qry'], transform_dic['pos'], transform_dic['neg']
        self.pos_return_num = pos_return_num
        self.neg_return_num = neg_return_num
    def __getitem__(self, idx):
        rslt_dic = super(TripletImageDataset, self).__getitem__(idx)
        qry, pos_lst, neg_lst, pos_pol, neg_pol = rslt_dic['qry'], rslt_dic['pos'], rslt_dic['neg'], rslt_dic['pos_policy'], rslt_dic['neg_policy']
        try:
            pos = random.sample(pos_lst, self.pos_return_num)
        except:
            raise Exception(f'pos_return_num should be smaller than length of positive list')
        try:
            neg = random.sample(neg_lst, self.neg_return_num)
        except:
            raise Exception(f'neg_return_num should be smaller than length of negative list')
        cat, sketch, prod = self.classify(self.get_basepath(qry))
        if self.load_images:
            qry_rslt = self.image_lst_im[qry]
            pos_rslt = [self.sketch_lst_im[i] for i in pos]
            neg_rslt = [self.sketch_lst_im[i] for i in neg]
        else:
            qry_rslt = Image.open(qry).convert('RGB')
            pos_rslt = [Image.open(i).convert('RGB') for i in pos]
            neg_rslt = [Image.open(i).convert('RGB') for i in neg]
        if self.transform_dic:
            qry_rslt = self.qry_trans(qry_rslt)
            pos_rslt = [self.pos_trans(i) for i in pos_rslt]
            neg_rslt = [self.neg_trans(i) for i in neg_rslt]
        else:
            qry_rslt = np.array(qry_rslt)
            pos_rslt = [np.array(i) for i in pos_rslt]
            neg_rslt = [np.array(i) for i in neg_rslt]
        return {'qry': qry_rslt, 'pos': pos_rslt, 'neg': neg_rslt, 'cat_idx': self.cat_idx[cat], 'prod_idx': self.prod_idx[prod]}
