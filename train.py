import os, argparse, yaml
import urllib.request
from types import SimpleNamespace
from urllib.error import HTTPError
from datetime import datetime
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import *
import torch.utils.data as data
import torchvision
from IPython.display import HTML, display, set_matplotlib_formats
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import timm
from collections import OrderedDict as OD
from collections import namedtuple as NT
from softdataset import TripletImageDataset
import pickle
from tqdm import tqdm
from torchvision.transforms import ToTensor, Resize
import torch.nn.functional as F
import torchvision.transforms.functional as FF
import AutoAugment

def run(args):
    
    model_dict = {}
    sp = args.save_path
    bs = args.batch_size
    expdir = args.expdir
    device = args.device
    path = args.ims_path
    inp_size = args.input_size
    model_name=args.model_name
    optimizer_name=args.optimizer_name
    lr = args.learning_rate
    wd = args.weight_decay
    checkpoint_path = args.checkpoint_path
    only_features = args.only_feature_embeddings
    only_labels = args.only_target_labels
    
    
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(f"\nTraining Arguments:\n{argstr}\n")
    
    optimizer_hparams={"lr": lr, "weight_decay": wd}
    model_dict[model_name] = 0       
      
    
    transformations = {}  
    # qry, pos, neg 서로 다른 transform을 적용 하기 위함  
        
    transformations['qry'] = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        AutoAugment.ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  
    transformations['pos'] = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  
    transformations['neg'] = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
    
    dataset = TripletImageDataset(data_dir=path, transform_dic=transformations)
    num_classes = dataset.get_prod_length()
    print(f"The dataset has {num_classes} classes")
    
    tr_ds, dataset_eval_copy = torch.utils.data.random_split(dataset, [int(len(dataset)*.8), len(dataset)-int(len(dataset)*.8)])
    val_ds, test_ds = torch.utils.data.random_split(dataset_eval_copy, [int(len(dataset_eval_copy)*.5), len(dataset_eval_copy)-int(len(dataset_eval_copy)*.5)])
    
    print(f"Number of train set images: {len(tr_ds)}")
    print(f"Number of validation set images: {len(val_ds)}")
    print(f"Number of test set images: {len(test_ds)}")
    
    with open(f'data/{datetime.now().strftime("%Y%m%d-%H%M%S")}-test_ds.pickle', 'wb') as handle:
        pickle.dump(test_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    cos = CosineSimilarity(dim=1, eps=1e-6)
    train_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=8)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=8)  
    labels = {"pos": torch.tensor(1.).unsqueeze(0),
              "neg": torch.tensor(-1.).unsqueeze(0)}
    
    assert only_features or only_labels, "Please choose at least one loss function to train the model (triplet loss or crossentropy loss)"
    if only_features and only_labels:
        print("\nTrain using triplet loss and crossentropy loss\n")
    elif only_features == True and only_labels == None:
        print("\nTrain using only triplet loss\n")                
    elif only_features == None and only_labels == True:
        print("\nTrain using only crossentropy loss\n")  
    
    class ContrastiveLoss(Module):

        """

        Contrastive loss

        Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise

        """

        def __init__(self, margin):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin
            self.eps = 1e-9

        def forward(self, output1, output2, target, size_average=True):
            distances = (output2 - output1).pow(2).sum(1)  # squared distances
            losses = 0.5 * (target * distances + (1 + -1 * target) * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
            return losses.mean() if size_average else losses.sum() 
    
    
    class Model(pl.LightningModule):

        def __init__(self, model_name,  optimizer_name, optimizer_hparams):
            """
            Gets model name, optimizer name and hparams and returns trained model (pytorch lightning) with results (dict).
            
            Arguments:
                model_name - Name of the model/CNN to run. Used for creating the model (see function below)
                optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
                optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
            """
            super().__init__()
            # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
            self.save_hyperparameters()

            # Create model
            self.model = create_model(model_name)
            # Create loss module
            # self.loss_module = ripletMarginLoss(margin=1.0, p=2).to('cuda')
            # self.loss_module = TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)).to('cuda')
            # self.loss_module = ContrastiveLoss(margin=0.2).to('cuda')
            self.cos_loss = CosineEmbeddingLoss(margin=0.5).to('cuda')
            self.ce_loss = CrossEntropyLoss()
            # Example input for visualizing the graph in Tensorboard
            self.example_input_array = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

        def forward(self, inp):
            
            # Function to convert dictionary to namedtuple
            def dict_to_namedtuple(dic):
                return NT('GenericDict', dic.keys())(**dic)
            
            dic = {}                        
            fm = self.model.forward_features(inp)
            pool = AvgPool2d((7,7))
            lbl = self.model.forward_head(fm)
            dic["feature_map"] = torch.reshape(pool(fm), (-1, fm.shape[1]))
            dic["class_pred"] = lbl
            out = dict_to_namedtuple(dic)
            
            return out
        
        def configure_optimizers(self):
            # self.hparams['lr'] = self.hparams.optimizer_hparams['lr']
            if self.hparams.optimizer_name == "Adam":
                # AdamW is Adam with a correct implementation of weight decay (see here
                # for details: https://arxiv.org/pdf/1711.05101.pdf)
                optimizer = torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
                # scheduler = {"scheduler": ReduceLROnPlateau(optimizer, verbose=True),
                # "monitor": "val_loss"}
            elif self.hparams.optimizer_name == "SGD":
                optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
            else:
                assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'
            
            scheduler = MultiStepLR(optimizer=optimizer, milestones=[10,20,30,40, 50], gamma=0.1, verbose=True)
        
            return [optimizer], [scheduler]
        
        def training_step(self, batch, batch_idx): # triplet loss 
            # "batch" is the output of the training data loader.
            
            cos_sims = []
            ims, poss, negs, clss, regs = batch['qry'], batch['pos'][0], batch['neg'][0], batch['cat_idx'], batch['prod_idx']

            # Get feature maps and pred labels
            out_ims = self(ims)
            fm_ims, lbl_ims = out_ims[0], out_ims[1] # get feature maps [0] and predicted labels [1]
            out_poss = self(poss)
            fm_poss, lbl_poss = out_poss[0], out_poss[1] # get feature maps [0] and predicted labels [1]
            out_negs = self(negs)
            fm_negs, lbl_negs = out_negs[0], out_negs[1] # get feature maps [0] and predicted labels [1]
            
            # Compute loss
            if only_features and only_labels:
                loss_cos = self.cos_loss(fm_ims, fm_poss, labels["pos"].to("cuda")) + self.cos_loss(fm_ims, fm_negs, labels["neg"].to("cuda"))
                loss_ce = self.ce_loss(lbl_ims, regs) + self.ce_loss(lbl_poss, regs)
                loss = loss_cos + loss_ce 
            elif only_features == True and only_labels == None:
                loss_cos = self.cos_loss(fm_ims, fm_poss, labels["pos"].to("cuda")) + self.cos_loss(fm_ims, fm_negs, labels["neg"].to("cuda"))
                loss = loss_cos                 
            elif only_features == None and only_labels == True:
                loss_ce = self.ce_loss(lbl_ims, regs) + self.ce_loss(lbl_poss, regs)
                loss = loss_ce 
                
            # Compute top3 and top1
            top3, top1 = 0, 0            
            for idx, fm in (enumerate(fm_ims)):
                sim = cos(fm.unsqueeze(0), fm_poss) 
                cos_sims.append(torch.mean(sim))
                vals, inds = torch.topk(cos(fm.unsqueeze(0), fm_poss), k=3)
                top3 += len(inds[idx == inds])
                top1 += len(inds[idx == inds[0]])


            # Logs the loss per epoch to tensorboard (weighted average over batches)
#             self.log("train_cos_loss", loss_cos)
#             self.log("train_ce_loss", loss_ce)
            self.log("train_loss", loss)
            self.log("train_top3", top3)
            self.log("train_top1", top1)

            return OD([('loss', loss)]) #, ('train_top3', top3 / len(ims)), ('train_top1', top1 / len(ims))])  # Return tensor to call ".backward" on

        def validation_step(self, batch, batch_idx): # triplet loss 

            cos_sims = []
            ims, poss, negs, clss, regs = batch['qry'], batch['pos'][0], batch['neg'][0], batch['cat_idx'], batch['prod_idx']

            # Get feature maps and pred labels
            out_ims = self(ims)
            fm_ims, lbl_ims = out_ims[0], out_ims[1] # get feature maps [0] and predicted labels [1]
            out_poss = self(poss)
            fm_poss, lbl_poss = out_poss[0], out_poss[1] # get feature maps [0] and predicted labels [1]
            out_negs = self(negs)
            fm_negs, lbl_negs = out_negs[0], out_negs[1] # get feature maps [0] and predicted labels [1]
            
            # Compute loss
            if only_features and only_labels:                
                loss_cos = self.cos_loss(fm_ims, fm_poss, labels["pos"].to("cuda")) + self.cos_loss(fm_ims, fm_negs, labels["neg"].to("cuda"))
                loss_ce = self.ce_loss(lbl_ims, regs) + self.ce_loss(lbl_poss, regs)
                loss = loss_cos + loss_ce 
            elif only_features == True and only_labels == None:
                loss_cos = self.cos_loss(fm_ims, fm_poss, labels["pos"].to("cuda")) + self.cos_loss(fm_ims, fm_negs, labels["neg"].to("cuda"))
                loss = loss_cos                 
            elif only_features == None and only_labels == True:
                loss_ce = self.ce_loss(lbl_ims, regs) + self.ce_loss(lbl_poss, regs)
                loss = loss_ce 
            
            # Compute top3 and top1            
            top3, top1 = 0, 0
            
            for idx, fm in (enumerate(fm_ims)):
                sim = cos(fm.unsqueeze(0), fm_poss) 
                cos_sims.append(torch.mean(sim))
                vals, inds = torch.topk(cos(fm.unsqueeze(0), fm_poss), k=3)
                top3 += len(inds[idx == inds])
                top1 += len(inds[idx == inds[0]])

            print(f"Total loss: {loss:.3f}")
#             print(f"Crossentropy loss: {loss_ce:.3f}")
            # print(f"Val loss: {loss:.3f}")
            # print(f"Val top3: {top3}")
            # print(f"Val top1: {top1}")
            # print(self.hparams.optimizer_hparams['lr'])
            # print(f"Similarity score: {torch.mean(torch.FloatTensor(cos_sims))}")

            # Logs the loss per epoch to tensorboard (weighted average over batches)
            self.log("val_loss", loss)
#             self.log("val_loss_cos", loss_cos)
#             self.log("val_loss_ce", loss_ce)
            self.log("cos_sims", torch.mean(torch.FloatTensor(cos_sims)))
            self.log("val_top3", top3)
            self.log("val_top1", top1)

            # return OD([('loss', loss), ('val_top3', top3), ('val_top1', top1)]) 
            return OD([('loss', loss), ('val_top3', top3),
                       ('cos_sims', torch.mean(torch.FloatTensor(cos_sims)))])

        def test_step(self, batch, batch_idx): # triplet loss 
            
            """ 
            Compares test images in the batch with the all images in the dataloader.
            
            """
            fms_ims, fms_poss, fms_negs, scores = [], [], [], []
            top1, top3, = 0, 0
            
            print("Obtaining embeddings...")
            # Get feature maps and pred labels of the whole test data
            for i, batch_all in enumerate(test_loader):
                ims_all, poss_all, negs_all, clss_all, regs_all = batch_all['qry'], batch_all['pos'][0], batch_all['neg'][0], batch_all['cat_idx'], batch_all['prod_idx']
                out_ims_all = self(ims_all.cuda())
                fm_ims_all, lbl_ims_all = out_ims_all[0], out_ims_all[1] # get feature maps [0] and predicted labels [1]
                out_poss_all = self(poss_all.cuda())
                fm_poss_all, lbl_poss_all = out_poss_all[0], out_poss_all[1] # get feature maps [0] and predicted labels [1]
                out_negs_all = self(negs_all.cuda())
                fm_negs_all, lbl_negs_all = out_negs_all[0], out_negs_all[1] # get feature maps [0] and predicted labels [1]
                fms_ims.extend(fm_ims_all)
                fms_poss.extend(fm_poss_all)
                fms_negs.extend(fm_negs_all)
            print("Embeddings are obtained!")               
            fms_ims = torch.stack(fms_ims)
            fms_poss = torch.stack(fms_poss)
            fms_negs = torch.stack(fms_negs)
            
            ims, poss, negs, clss, regs = batch['qry'], batch['pos'][0], batch['neg'][0], batch['cat_idx'], batch['prod_idx']
            # Get feature maps and pred labels
            out_ims = self(ims)
            fm_ims, lbl_ims = out_ims[0], out_ims[1] # get feature maps [0] and predicted labels [1]
            out_poss = self(poss)
            fm_poss, lbl_poss = out_poss[0], out_poss[1] # get feature maps [0] and predicted labels [1]
            out_negs = self(negs)
            fm_negs, lbl_negs = out_negs[0], out_negs[1] # get feature maps [0] and predicted labels [1]
            
             # Compute loss
            if only_features and only_labels:
                loss_cos = self.cos_loss(fm_ims, fm_poss, labels["pos"].to("cuda")) + self.cos_loss(fm_ims, fm_negs, labels["neg"].to("cuda"))
                loss_ce = self.ce_loss(lbl_ims, regs) + self.ce_loss(lbl_poss, regs)
                loss = loss_cos + loss_ce 
            elif only_features == True and only_labels == None:
                loss_cos = self.cos_loss(fm_ims, fm_poss, labels["pos"].to("cuda")) + self.cos_loss(fm_ims, fm_negs, labels["neg"].to("cuda"))
                loss = loss_cos                 
            elif only_features == None and only_labels == True:
                loss_ce = self.ce_loss(lbl_ims, regs) + self.ce_loss(lbl_poss, regs)
                loss = loss_ce 
        
            # Compute top3 and top1   
            for index, fm in enumerate(fm_ims):
                score = cos(fm.unsqueeze(0), fms_ims) # (1, fm), (len(dl), fm) = (len(dl), fm)
                scores.append(torch.mean(score))
                vals, inds = torch.topk(cos(fm.unsqueeze(0), fms_ims), k=3)
                top3 += len(inds[index == inds])
                top1 += len(inds[index == inds[0]])     

            print(f'scores: {torch.mean(torch.FloatTensor(scores)).item():.3f}')
            print('top1', top1/len(fm_ims))
            print('top3', top3/len(fm_ims))
            print('test_loss', loss / len(fm_ims))

            self.log("scores", torch.mean(torch.FloatTensor(scores)).item())
            self.log("test_loss", loss / len(fm_ims))
            self.log("test_top3", top3 / len(fm_ims))
            self.log("test_top1", top1 / len(fm_ims))

    def create_model(model_name, conv_input=False, num_classes=num_classes):
        
        """ 
        
        Gets model name and creates a timm model.
        
        """

        if model_name in model_dict:
            base_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
            print(f"Model {model_name} with the best weights is successfully loaded!")        
            if conv_input:                
                conv_layer = Sequential(Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1),padding=(1,1), bias=False), 
                 # Conv2d(3, 3, kernel_size=(3, 3), stride=(2, 2),padding=(1,1), bias=False), 
                 # MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                 SiLU(inplace=True))
                model = Sequential(conv_layer, base_model)  
            else:
                model = base_model
        else:
            assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'
            
        return model

    def train_model(model_name, save_name=None, **kwargs):
        
        """
        Trains the model and returns trained model with its results.
        
        Arguments:
            model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
            save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
        """
        if save_name is None:
            save_name = model_name

        # Create a PyTorch Lightning trainer with the generation callback
        trainer = pl.Trainer(
            default_root_dir=os.path.join(sp, save_name),  # dir name to save models
            # Run on a single GPU (if possible)
            gpus=1 if str(device) == "cuda:0" else 0,
            precision=16, amp_backend='native',
            # total num of epochs
            max_epochs=300,
            log_every_n_steps=15,
            # auto_lr_find=True,
#             fast_dev_run=True,
            strategy="ddp", accelerator="gpu", devices=3,
            callbacks=[
                
                ModelCheckpoint(
                    filename='{epoch}-{val_loss:.2f}-{cos_sims:.2f}-{val_top3:.2f}', 
                    every_n_train_steps = None, save_top_k=1,
                    save_weights_only=True, mode="max", monitor="val_top1" 
                ),  # Save the best checkpoint based on the min val_loss recorded. Saves only weights and not optimizer
                EarlyStopping(monitor="val_top1", mode="max", patience=10, verbose=True), # set the metric (and change the mode!) to track for early stopping
                LearningRateMonitor("epoch"), # Log learning rate every epoch
            ]
        )
        trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filename = os.path.join(sp, 'models', '.ckpt')
        if os.path.isfile(pretrained_filename):
            print(f"Found pretrained model at {pretrained_filename}, loading...")
            # Automatically loads the model with the saved hyperparameters
            model = Model.load_from_checkpoint(pretrained_filename)
        else:
            pl.seed_everything(42)  # To be reproducable
            model = Model(model_name=model_name, **kwargs)
            # lr_finder = trainer.tuner.lr_find(model)
            # model.hparams.learning_rate = lr_finder.suggestion()
            trainer.fit(model, train_loader, val_loader)
#             model = Model.load_from_checkpoint(os.path.join(trainer.checkpoint_callback.best_model_path, 'blabla')) # load best checkpoint after training

        # Test best model on validation and test set
        test_result = trainer.test(model, dataloaders=test_loader, verbose=True)

        result = {"test_loss": test_result[0]["test_loss"], 
                  "test_scores": test_result[0]["scores"],
                  "test_top3": test_result[0]["test_top3"],
                  "test_top1": test_result[0]["test_top1"]}

        return model, result    
    
    trained_model, results = train_model(
    model_name=model_name, optimizer_name=optimizer_name, save_name=f"{model_name}_{optimizer_name}_{lr}",
    optimizer_hparams=optimizer_hparams)
    test_loss = results['test_loss']
    test_top1 = results['test_top1']
    test_top3 = results['test_top3']
    with open(f"results/{model_name}_{optimizer_name}_{lr}_{test_loss}_{test_top1}_{test_top3}_results.pickle", 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results of the training are saved in results/{model_name}_{optimizer_name}_{lr}_{test_loss}_{test_top1}_{test_top3}_results.pickle")   

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Triplet Loss PyTorch Lightning Arguments')
    parser.add_argument('-ed', '--expdir', default=None, help='Experiment directory')
    parser.add_argument("-sp", "--save_path", type=str, default='saved_models', help="Path to save trained models")
    parser.add_argument('-cp', '--checkpoint_path', type=str, default="/home/ubuntu/workspace/bekhzod/triplet-loss-pytorch/pytorch_lightning/saved_models/model_best.pth.tar", help='Path to the trained model')
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-d", "--device", type=str, default='cuda:1', help="GPU device number")
#     parser.add_argument("-ip", "--ims_path", type=str, default='/home/ubuntu/workspace/dataset/test_dataset_svg', help="Path to the images")
    parser.add_argument("-ip", "--ims_path", type=str, default='/mnt/test_dataset_svg/test_dataset_svg_1121', help="Path to the images")
    parser.add_argument("-is", "--input_size", type=int, default=(224, 224), help="Size of the images")
    parser.add_argument("-mn", "--model_name", type=str, default='rexnet_150', help="Model name (from timm library (ex. darknet53, ig_resnext101_32x32d))")
    parser.add_argument("-on", "--optimizer_name", type=str, default='Adam', help="Optimizer name (Adam or SGD)")
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4, help="Learning rate value")
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-5, help="Weight decay value")
    parser.add_argument("-ofm", "--only_feature_embeddings", type=bool,  
                        help="If True trains the model using only triplet loss and and return feature embeddings (if both otl and ofm are True uses two loss functions simultaneously)")
    parser.add_argument("-otl", "--only_target_labels", type=bool, 
                        help="If True trains the model using only cross entropy and and return predicted labels (if both otl and ofm are True uses two loss functions simultaneously)")
    
    args = parser.parse_args() 
    
    run(args) 
