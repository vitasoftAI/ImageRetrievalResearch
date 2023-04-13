# Import libraries
from torch.utils.data import Dataset
import glob, os, random, json, tqdm, csv
from PIL import Image
import numpy as np

def data_split(data_dir, out_path, policy: str = 'prod', hard_split: bool = True, train_essentials: str = '', split: list = [0.8, 0.1, 0.1]):
    
    """
    
    This function gets data directory, output path, policy for data split, hard_split note, essential classes for train, and split ratio and returns output path with the data split.
    
    Arguments:
    
        data_dir            - directory with the data, str;
        out_path            - path to output the json file with data split, str;
        policy              - policy to split data, str;
        hard_split          - if True, not all classes in the validation and test sets, else train, val, test sets have samples for all classes, bool;
        train_essentials    - csv file with class names for train, str;
        split               - data split ratio, list.
        
    Output:
    
        out_path            - path to output with the json file that has data split, str.
    
    """    
    
    assert sum(split) == 1, " Please, make sure that sum of split list equals to exactly 1"
    
    # Initialize train essentials list
    train_essential = []
    
    # Read csv file
    if train_essentials:
        with open(train_essentials, 'r') as f:
            data = csv.reader(f)
            for i in data: train_essential += i
    
    # Get image list
    lst = glob.glob(os.path.join(data_dir, '**/*'), recursive = True)
    lst = list(set(lst) - set(glob.glob(os.path.join(data_dir, '*/pdf_detail/*'))))
    
    # Get only files
    lst = [i for i in lst if os.path.isfile(i)]
    
    # Shuffle the list
    random.shuffle(lst)
    
    # Create dictionary with train and validation keys
    rslt = {'train':[], 'val':[]}
    
    # Create test key in split contains test
    if len(split) == 3: rslt['test'] = []
    dic = {}
    
    # Go through the list
    for i in lst:
        
        # Get path
        path = i.replace(os.path.join(data_dir, ''), '')
        split_path = path.split('/')
        
        # Get cat number and prod number
        cat, prod = split_path[0], split_path[1].split('_')[-2]
        if policy == 'cat': pol = cat
        elif policy == 'prod': pol = prod
        else: raise Exception('policy must be one of [cat, prod]')
        if pol not in dic: dic[pol] = []
        dic[pol].append(i)
    
    # Hard split
    if hard_split: keys = list(dic.keys())
        
        # Get train essentials
        train_essential = list(set(keys) & set(train_essential))
        keys = list(set(keys) - set(train_essential))
        random.shuffle(keys)
        train_idx, val_idx = int(len(keys)*split[0]), int(len(keys)*split[1])
        train_keys = keys[:train_idx] + train_essential
        val_keys = keys[train_idx: train_idx + val_idx]
        
        # Add test keys
        if len(split) == 3: test_keys = keys[train_idx + val_idx:]
        for key in train_keys: rslt['train'] += dic[key]
        for key in val_keys: rslt['val'] += dic[key]
        if len(split) == 3:
            for key in test_keys:
                rslt['test'] += dic[key]
        
        # Create json file and save it
        with open(out_path, 'w') as f:
            json.dump(rslt, f)
            
        return out_path
    
    # Soft split
    else:
        for key, value in dic.items():
            if key in train_essential: rslt['train'] += value
            else:
                idx = int(len(value) * split[1])
                val_len = max(idx, 1)
                test_len = max(int(len(value) * split[2]), 1)
                train_len = len(value) - val_len - test_len
                if val_len > 0 and test_len > 0 and train_len > 0:
                    rslt['val'] + =value[:val_len]
                    rslt['test'] += value[val_len:val_len + test_len]
                    rslt['train'] += value[val_len + test_len:]
                else:
                    rslt['val']+=value
                    rslt['test']+=value
                    rslt['train']+=value
        
        # Create json file and save it
        with open(out_path, 'w') as f: json.dump(rslt, f)
        return out_path

class OriginalDataset(Dataset):
    
    """
    
    This function gets several arguments and return a dataset object with data from the given directory path.
    
    Arguments:
    
        data_dir            - directory with the data, str;
        random              - option for random image extraction, bool;
        pos_policy          - type of the calling positive images from the folder, str;
        neg_policy          - type of the calling negative images from the folder, str;
        data_json           - path to the json file, str;
        trainval            - option for train or validation, str;
        
    Output:
    
        ds                  - dataset, torch dataset object.
    
    """    
    
    def __init__(self, data_dir, random = True, pos_policy = 'prod', neg_policy = 'except_cat', trainval_json = None, trainval = None, data_json = None):
    
        # Get dataset arguments
        self.pos_policy, self.neg_policy, self.random, self.data_dir = pos_policy, neg_policy, random, data_dir
        
        # Initialize dictionaries
        self.cat_idx, self.prod_idx, self.pos_neg_dic, self.neg_dic = {}, {}, {}, {}
        
        # Not random case
        if not self.random:
            assert data_json != None, "data_json is required if not random"
            assert trainval_json == None and trainval == None, 'random false mode doesn\t support trainval mode'
            
            # Read data from the json file 
            with open(data_json, 'r') as f: json_data = json.loads(f.read())
            
            # Get information from the json file
            self.cat_idx = json_data['meta']['cat_idx']
            self.prod_idx = json_data['meta']['prod_idx']
            self.sketch_lst = json_data['meta']['sketch_lst']
            self.image_lst = json_data['meta']['image_lst']
            self.data = json_data['data']
            
        # Random case
        else:
            # From a json file
            if trainval_json:
                assert trainval != None, "Please declare whether this is train or val dataset"
                with open(trainval_json, 'r') as f: trainval_data = json.loads(f.read())
                self.image_lst = trainval_data[trainval]
            
            # Not from a json file
            else: self.image_lst = glob.glob(os.path.join(self.data_dir, '**/*'), recursive=True)
            
            # Get image lists
            self.sketch_lst = glob.glob(os.path.join(self.data_dir, '*/pdf_detail/*'))
            self.image_lst = list(set(self.image_lst) - set(self.sketch_lst))
            self.image_lst = [i for i in self.image_lst if os.path.isfile(i)]
            self.sketch_lst = [i for i in self.sketch_lst if os.path.isfile(i)]
            
            # Initialize dictionaries
            self.cat_dic, self.prod_dic, self.sketch_dic = {}, {}, {}
            
            for i in self.image_lst:
                cat, prod = self.image_classify(i)
                self.cat_dic = self.gen_dic(self.cat_dic, i, cat, 'real')
                self.prod_dic = self.gen_dic(self.prod_dic, i, prod, 'real')
            for idx, key in enumerate(self.cat_dic.keys()):
                self.cat_idx[key] = idx
            for idx, key in enumerate(self.prod_dic.keys()):
                self.prod_idx[key] = idx
            for i in self.sketch_lst:
                cat, prod = self.sketch_classify(i)
                self.cat_dic = self.gen_dic(self.cat_dic, i, cat, 'sketch')
                self.prod_dic = self.gen_dic(self.prod_dic, i, prod, 'sketch')
            for qry in self.image_lst:
                cat, prod = self.image_classify(qry)
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
    
    This function gets an index and returns dictionary with data.
    
    Argument:
    
        idx     - index of the dataset, int.
        
    Output:
    
        dic     - dictionary with data, dictionary.
    
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
    
    def get_cat_length(self): return len(self.cat_idx)
    
    def get_prod_length(self): return len(self.prod_idx)
    
    def image_classify(self, path):
        
        path = self.get_basepath(path)
        split_path = path.split('/')
        cat, prod = split_path[0], split_path[1].split('_')[-2]
        
        return cat, prod
    
    def sketch_classify(self, path):
        
        path = self.get_basepath(path)
        split_path = path.split('/')
        cat, prod = split_path[0], split_path[2].split('_')[-2]
        
        return cat, prod
    
    def gen_dic(self, dic, f_name, parser, sketch_or_real):
        
        basepath = self.get_basepath(f_name)
        if parser not in dic:
            dic[parser] = {'sketch': [], 'real': []}
        dic[parser][sketch_or_real].append(f_name)
        
        return dic

class OriginalImageDataset(OriginalDataset):
    
    """
    
    This class gets several arguments and returns dictionary with data.
    
    Arguments:
    
        transform_dic  - transformations, dictionary;
        pos_return_num - number of images to be returned as a positive image, int;
        neg_return_num - number of images to be returned as a negative image, int;
        load_images    - option to load images or not, bool;
        
    Output:
    
        di             - a dictionary contaning input, positive, and negative images, category index, and product number index.
    
    """
    
    def __init__(self, transform_dic = None, pos_return_num = 1, neg_return_num = 1, load_images = False, **kwargs):
        super(OriginalImageDataset, self).__init__(**kwargs)
        self.load_images = load_images
        # Load images
        if self.load_images:
            self.sketch_lst_im = {i: Image.open(i).convert('RGB') for i in self.sketch_lst}
            self.image_lst_im = {i: Image.open(i).convert('RGB') for i in self.image_lst}
        
        # Get transformations dictionary
        self.transform_dic = transform_dic
        
        # Apply transformations
        if transform_dic:
            self.qry_trans, self.pos_trans, self.neg_trans = transform_dic['qry'], transform_dic['pos'], transform_dic['neg']
        
        # Get positive and negative images numbers
        self.pos_return_num = pos_return_num
        self.neg_return_num = neg_return_num
        
    def __getitem__(self, idx):
        
        """
        
        This function gets an index and returns data in a dictionary form.
        
        Argument:
        
            idx    - index in the dataset, int.
            
        Output:
        
            di     - dictionary containing necessary data for training, dictionary.
        
        """
        
        rslt_dic = super(OriginalImageDataset, self).__getitem__(idx)
        
        # Get images
        qry, pos_lst, neg_lst, pos_pol, neg_pol = rslt_dic['qry'], rslt_dic['pos'], rslt_dic['neg'], rslt_dic['pos_policy'], rslt_dic['neg_policy']
        
        try: pos = random.sample(pos_lst, self.pos_return_num)
        except: raise Exception(f'pos_return_num should be smaller than length of positive list')
        
        try: neg = random.sample(neg_lst, self.neg_return_num)
        except: raise Exception(f'neg_return_num should be smaller than length of negative list')
        
        # Get category and product numbers using query image
        cat, prod = self.image_classify(qry)
        
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
