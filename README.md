# TripletImageDatasetTraining

![Image](https://user-images.githubusercontent.com/50166164/202942637-e3674ee5-56ae-4ffb-830d-d5e42fb91072.PNG)

Run training 
```python
python train.py --batch_size=64 --optimizer_name="Adam" --only_feature_embeddings=True --only_target_labels=True 
```

### ROC curve from scratch

Computes true positive rate (TPR) based on true positives (TP) and false negatives (FN); False Positive Rate based on False Positives (FP) and True Negatives (TN).
Computes area under curve (AUC) score of receiver operating characteristics (ROC) curve and plots it.

![Image](https://user-images.githubusercontent.com/50166164/203878341-69123d1a-5c66-41ed-9870-1dc0dedc900e.PNG)

Run the code 
```python
python train.py 
```
