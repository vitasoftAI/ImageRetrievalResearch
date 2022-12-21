# Image Retrieval Research
### Retrieving sketch images given an input (query) image

### Create virtual environment
```python
conda create -n <ENV_NAME> python=3.9
conda activate <ENV_NAME>
pip install -r requirements.txt
```

### Data Split (Original Image Dataset)
```python
from original_dataset_downsampling import OriginalImageDataset, data_split

root = '/home/ubuntu/workspace/dataset/test_dataset_svg/spec72'
out_path = 'data/spec72.json'

data_split(root, out_path, hard_split=False)
```

### Get Data (Original Image Dataset)
```python
path = "~/1209_refined_data"

transformations = {}   
transformations['qry'] = transforms.Compose([
                        transforms.ToTensor()])
transformations['pos'] = transforms.Compose([transforms.ToTensor()])  
transformations['neg'] = transforms.Compose([transforms.ToTensor()])

out_path = "data/1209_refined_data.json"
tr_ds = OriginalImageDataset(data_dir = path, transform_dic=transformations, random=True, trainval_json=out_path, trainval='train', load_images=False)
val_ds = OriginalImageDataset(data_dir = path, transform_dic=transformations, random=True, trainval_json=out_path, trainval='val', load_images=False)
test_ds = OriginalImageDataset(data_dir = path, transform_dic=transformations, random=True, trainval_json=out_path, trainval='test', load_images=False)
```

### Run training 
```python
python train.py --batch_size=64 --optimizer_name="Adam" --lr=3e-4 --model_name="efficientnet_b3a"
```
![Image](https://user-images.githubusercontent.com/50166164/202942637-e3674ee5-56ae-4ffb-830d-d5e42fb91072.PNG)

### Pre-trained checkpoints on Sketchy Database: see checkpoints directory

### ROC curve from scratch

Computes true positive rate (TPR) based on true positives (TP) and false negatives (FN); False Positive Rate based on False Positives (FP) and True Negatives (TN).
Computes area under curve (AUC) score of receiver operating characteristics (ROC) curve and plots it.

![Image](https://user-images.githubusercontent.com/50166164/203878341-69123d1a-5c66-41ed-9870-1dc0dedc900e.PNG)

Run using terminal
```python
python roc_curve_from_scratch.py
```

### Image Retrieval Model Results

![Capture](https://user-images.githubusercontent.com/50166164/208335211-055d9140-b79f-4594-9fb8-632cfcec409d.PNG)
![Capture1](https://user-images.githubusercontent.com/50166164/208335229-8304f5ea-2dc3-448c-b7a0-708e2c3a68b0.PNG)
![Capture2](https://user-images.githubusercontent.com/50166164/208335217-b67652b7-e079-4866-8291-e3ec26d29434.PNG)
![Capture3](https://user-images.githubusercontent.com/50166164/208335238-3d67924d-409d-4436-8551-0e08e0be978b.PNG)
![Capture4](https://user-images.githubusercontent.com/50166164/208335241-0fcc0f7f-f6a7-431a-ac2b-0cae10e5f82a.PNG)
![Capture5](https://user-images.githubusercontent.com/50166164/208335243-2bf5fdcb-7723-45d1-ac77-d5c1c89f972b.PNG)
