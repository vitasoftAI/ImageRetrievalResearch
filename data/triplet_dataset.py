import os
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from glob import glob


def find_classes(root):
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_idex = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idex


def make_dataset(root):
    images = []

    cnames = os.listdir(root)
    for cname in cnames:
        c_path = os.path.join(root, cname)
        if os.path.isdir(c_path):
            fnames = os.listdir(c_path)
            for fname in fnames:
                path = os.path.join(c_path, fname)
                images.append(path)

    return images


class TripleDataset(data.Dataset):
    def __init__(self, photo_root, sketch_root, transform=None):
        super(TripleDataset, self).__init__()
        self.transform = transform
        classes, class_to_idx = find_classes(photo_root)

        self.photo_root = photo_root
        self.sketch_root = sketch_root

        self.photo_paths = sorted(make_dataset(self.photo_root))
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.len = len(self.photo_paths)

    def __getitem__(self, index):

        photo_path = self.photo_paths[index]
        sketch_path, neg, label = self._getrelate_sketch(photo_path)

        photo = Image.open(photo_path).convert('RGB')
        sketch = Image.open(sketch_path).convert('RGB')
        neg = Image.open(neg).convert('RGB')
        P = photo
        S = sketch
        N = neg
        if self.transform is not None:
            P = self.transform(photo)
            S = self.transform(sketch)
            N = self.transform(neg)
        L = label
        return {'P': P, 'S': S, 'N':N, 'L': L}

    def __len__(self):
        return self.len

    def _getrelate_sketch(self, photo_path):

        
        cname = os.path.basename(os.path.dirname(photo_path))

        label = self.class_to_idx[cname]
        sketchs = sorted(os.listdir(os.path.join(self.sketch_root, cname)))
        
        items = set(os.listdir(self.sketch_root)) - set(cname)
        rnd = np.random.randint(0, len(items))
        neg = os.path.join(self.sketch_root, list(items)[rnd])
        files = glob(os.path.join(neg, '*'))
        rnd = np.random.randint(0, len(files))
        neg = files[rnd]
        
        
        sketch_rel = []
        for sketch_name in sketchs:
            sketch_rel.append(sketch_name)

        rnd = np.random.randint(0, len(sketch_rel))

        sketch = sketch_rel[rnd]

        return os.path.join(self.sketch_root, cname, sketch), neg, label
