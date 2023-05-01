# Import libraries
from torch.utils.data import Dataset
import glob, os, random, tqdm, json, numpy as np
from PIL import Image

def data_split(data_dir, out_path, policy: str = 'cat', hard_split: bool = True, train_essentials: str = '', split: list = [0.8, 0.1, 0.1], sketch_qry = False):
    
    """
    
    This function gets data directory, output path, policy for data split, hard_split note, essential classes for train, and split ratio and returns output path with the data split.
    
    Arguments:
    
        data_dir            - directory with the data, str;
        out_path            - path to output the json file with data split, str;
        policy              - policy to split data, str;
        hard_split          - if True, not all classes in the validation and test sets, else train, val, test sets have samples for all classes, bool;
        train_essentials    - csv file with class names for train, str;
        split               - data split ratio, list;
        sketch_qry          - whether a sketch image can be used as a query image or not, bool.
        
    Output:
    
        out_path            - path to output with the json file that has data split, str.
    
    """    
    
    assert sum(split) == 1, 'sum of split should be 1'
    train_essential = []
    if train_essentials:
        with open(train_essentials, 'r') as f:
            data = csv.reader(f)
            for i in data:
                train_essential+=i
    if sketch_qry:
        lst = glob.glob(os.path.join(data_dir, 'photo/tx_000000000000/*/*'), recursive=True) + glob.glob(os.path.join(data_dir, 'sketch/tx_000000000000/*/*'), recursive=True)
    else:
        lst = glob.glob(os.path.join(data_dir, 'photo/tx_000000000000/*/*'), recursive=True)
    lst = [i for i in lst if os.path.isfile(i)]
    random.shuffle(lst)
    rslt = {'train':[], 'val':[]}
    if len(split) == 3:
        rslt['test'] = []
    dic = {}
    for i in lst:
        basename = os.path.basename(i)
        cat, prod = os.path.basename(os.path.dirname(i)), basename.split('-')[0].replace('.jpg', '')
        if policy == 'cat':
            pol = cat
        elif policy == 'prod':
            pol = prod
        else:
            raise Exception('policy must be one of [cat, prod]')
        if pol not in dic:
            dic[pol] = []
        dic[pol].append(i)
    if hard_split:
        keys = list(dic.keys())
        train_essential = list(set(keys) & set(train_essential))
        keys = list(set(keys) - set(train_essential))
        random.shuffle(keys)
        train_idx, val_idx = int(len(keys)*split[0]), int(len(keys)*split[1])
        train_keys = keys[:train_idx] + train_essential
        val_keys = keys[train_idx: train_idx + val_idx]

        if len(split) == 3:
            test_keys = keys[train_idx + val_idx:]
        for key in train_keys:
            rslt['train'] += dic[key]
        for key in val_keys:
            rslt['val'] += dic[key]
        if len(split) == 3:
            for key in test_keys:
                rslt['test'] += dic[key]
        with open(out_path, 'w') as f:
            json.dump(rslt, f)
        return out_path
    
    else:
        for key, value in dic.items():
            if key in train_essential:
                rslt['train']+=value
            else:
                idx = int(len(value)*split[1])
                val_len = max(idx, 1)
                test_len = max(int(len(value)*split[2]), 1)
                train_len = len(value) - val_len - test_len
                if val_len > 0 and test_len > 0 and train_len > 0:
                    rslt['val'] += value[:val_len]
                    rslt['test'] += value[val_len:val_len + test_len]
                    rslt['train'] += value[val_len+test_len:]
                else:
                    rslt['val'] += value
                    rslt['test'] += value
                    rslt['train'] += value
        with open(out_path, 'w') as f: json.dump(rslt, f)
        return out_path

class SketchyDataset(Dataset):

"""

This class gets several parameters and returns Sketchy Dataset.

Parameters:

    data_dir          - path to directory with data, str
    random            - option for random data extraction, bool;
    pos_policy        - policy for positive image selection, bool;
    neg_policy        - policy for negative image selection
    trainval          - option for train or validation, str;
    data_json         - json data file, json;
    sketch_qry        - whether a sketch image can be query or not, bool.

"""

    def __init__(self, data_dir, random = True, pos_policy = 'cat', neg_policy = 'except_cat', trainval_json = None, trainval = None, data_json = None, sketch_qry = False):
        self.pos_policy, self.neg_policy, self.random, self.data_dir = pos_policy, neg_policy, random, data_dir
        self.neg_dic, self.pos_neg_dic, self.prod_idx, self.cat_idx = {}, {}, {}, {}
        
        # When random option is off
        if not self.random:
            assert data_json != None, 'data_json is required if not random'
            assert trainval_json == None and trainval == None, 'random false mode doesn\t support trainval mode'
            with open(data_json, 'r') as f: json_data = json.loads(f.read())
            self.cat_idx = json_data['meta']['cat_idx']
            self.prod_idx = json_data['meta']['prod_idx']
            self.sketch_lst = json_data['meta']['sketch_lst']
            self.image_lst = json_data['meta']['image_lst']
            self.data = json_data['data']
        
        # When random option is on
        else:
            if trainval_json:
                assert trainval != None, 'you should declare whether this is train or val dataset'
                with open(trainval_json, 'r') as f:
                    trainval_data = json.loads(f.read())
                self.image_lst = trainval_data[trainval]
            else:
                self.image_lst = glob.glob(os.path.join(self.data_dir, 'photo/tx_000000000000/*/*'))
            
            self.sketch_lst = glob.glob(os.path.join(self.data_dir, 'sketch/tx_000000000000/*/*'))
            self.image_lst = [i for i in self.image_lst if os.path.isfile(i)]
            self.sketch_lst = [i for i in self.sketch_lst if os.path.isfile(i)]
            self.cat_dic, self.prod_dic = {}, {}
            for i in self.sketch_lst+self.image_lst:
                basepath = self.get_basepath(i)
                cat, prod = self.classify(basepath)
                self.cat_dic = self.gen_dic(self.cat_dic, i, cat)
                self.prod_dic = self.gen_dic(self.prod_dic, i, prod)
                # self.sketch_dic = self.gen_dic(self.sketch_dic, i, sketch_name)
            for idx, key in enumerate(self.cat_dic.keys()):
                self.cat_idx[key] = idx
            for idx, key in enumerate(self.prod_dic.keys()):
                self.prod_idx[key] = idx
                
            if sketch_qry:
                self.image_lst = self.image_lst + self.sketch_lst
            for qry in self.image_lst:
                basepath = self.get_basepath(qry)
                cat, prod = self.classify(basepath)
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
        
        """
        
        This function gets index and returns a dictionary with necessary information.
        
        Parameter:
        
            idx    - index, int.
            
        Output:
        
            dic    - information necessary for training, dictionary. 
        
        """
        
        if not self.random: return self.data[idx]
        qry = self.image_lst[idx]
        pos_neg = self.pos_neg_dic[self.image_lst[idx]]
        pos_lst, neg_lst, pos_policy, neg_policy = pos_neg['pos'], pos_neg['neg'], pos_neg['pos_policy'], pos_neg['neg_policy']
        
        return {'qry':qry, 'pos':pos_neg['pos'], 'neg':pos_neg['neg'], 'pos_policy': pos_neg['pos_policy'], 'neg_policy': pos_neg['neg_policy']}
    
    def __len__(self):
        if not self.random:
            return len(self.data)
        return len(self.image_lst)
    
    def get_basepath(self, path): return path.replace(os.path.join(self.data_dir, ''), '')
    
    def classify(self, path):
        basename = os.path.basename(path)
        cat, prod = os.path.basename(os.path.dirname(path)), basename.split('-')[0].replace('.jpg', '')
        return cat, prod
    
    def get_cat_length(self): return len(self.cat_dic)
    def get_prod_length(self): return len(self.prod_dic)
    
    def gen_dic(self, dic, f_name, parser):
        basepath = self.get_basepath(f_name)
        if parser not in dic:
            dic[parser] = {'sketch': [], 'photo': []}
        dic[parser][basepath.split('/')[0]].append(f_name)
        
        return dic
    
class SketchyImageDataset(SketchyDataset):
    
    
    """
    
    This class gets several parameters and returns a sketchy dataset.
    
    Parameters:
    
        transform_dic      - transformations to be applied, dict;
        pos_return_num     - number of images to be returned as a positive image, int;
        neg_return_num     - number of images to be returned as a negatuve image, int;
        load_images        - whether or not to load images, bool.
        
    Output:
    
        dic                - output of the class with necessary information, dict.
    
    """
    
    def __init__(self, transform_dic = None, pos_return_num = 1, neg_return_num = 1, load_images = False, **kwargs):
        super(SketchyImageDataset, self).__init__(**kwargs)
        self.load_images, self.transform_dic = load_images, transform_dic
        if self.load_images:
            self.sketch_lst_im = {i: Image.open(i).convert('RGB') for i in self.sketch_lst}
            self.image_lst_im = {i: Image.open(i).convert('RGB') for i in self.image_lst}
        
        if transform_dic: self.qry_trans, self.pos_trans, self.neg_trans = transform_dic['qry'], transform_dic['pos'], transform_dic['neg']
        
        self.pos_return_num, self.neg_return_num = pos_return_num, neg_return_num
        
    def __getitem__(self, idx):
        
        """
        
        This function gets index and returns a dictionary with necessary information.
        
        Parameter:
        
            idx    - index, int.
            
        Output:
        
            dic    - information necessary for training, dictionary. 
        
        """
        
        rslt_dic = super(SketchyImageDataset, self).__getitem__(idx)
        qry, pos_lst, neg_lst, pos_pol, neg_pol = rslt_dic['qry'], rslt_dic['pos'], rslt_dic['neg'], rslt_dic['pos_policy'], rslt_dic['neg_policy']
        try: pos = random.sample(pos_lst, self.pos_return_num)
        except: raise Exception(f'pos_return_num should be smaller than length of positive list')
        try: neg = random.sample(neg_lst, self.neg_return_num)
        except: raise Exception(f'neg_return_num should be smaller than length of negative list')
        
        cat, prod = self.classify(self.get_basepath(qry))
        
        if self.load_images: qry_rslt, pos_rslt, neg_rslt = self.image_lst_im[qry], [self.sketch_lst_im[i] for i in pos], [self.sketch_lst_im[i] for i in neg]
            
        else: qry_rslt, pos_rslt, neg_rslt = Image.open(qry).convert('RGB'), [Image.open(i).convert('RGB') for i in pos], [Image.open(i).convert('RGB') for i in neg]
             
        if self.transform_dic: qry_rslt, pos_rslt, neg_rslt = self.qry_trans(qry_rslt), [self.pos_trans(i) for i in pos_rslt], [self.neg_trans(i) for i in neg_rslt]
             
        else: qry_rslt, pos_rslt, neg_rslt = np.array(qry_rslt), [np.array(i) for i in pos_rslt], [np.array(i) for i in neg_rslt]
             
        return {'qry': qry_rslt, 'pos': pos_rslt, 'neg': neg_rslt, 'cat_idx': self.cat_idx[cat], 'prod_idx': self.prod_idx[prod]}
