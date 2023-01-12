import os, argparse, yaml, torch, torchvision, timm, pickle, wandb
from datetime import datetime
import pytorch_lightning as pl
from torch.nn import *
import torch.utils.data as data
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from collections import OrderedDict as OD
from collections import namedtuple as NT
from softdataset import TripletImageDataset
from original_dataset import OriginalImageDataset, data_split
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import AutoAugment
from dataset import TripleDataset


def run(args):
    
    # Get train arguments
    sp = args.save_path
    bs = args.batch_size
    expdir = args.expdir
    device = args.device
    path = args.ims_path
    model_name=args.model_name
    optimizer_name=args.optimizer_name
    lr = args.learning_rate
    wd = args.weight_decay
    only_features = args.only_feature_embeddings
    only_labels = args.only_target_labels
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(f"\nTraining Arguments:\n{argstr}\n")
    
    # Initialize train dictionaries:
    # for model;
    model_dict = {}    
    model_dict[model_name] = 0 
    # for optimizer;
    optimizer_hparams={"lr": lr, "weight_decay": wd}
    
    # Initialize transformations
    t = transforms.Compose([transforms.ToTensor()])
    
    # Get dataset
    ds = ImageFolder(path, transform = t)
    # Get length of the dataset
    len_ds = len(ds)
    print(len(ds))
    
    # Get number of classes
    num_classes = len(ds.classes)
    
    # Split the dataset into train and validation
    tr_ds, val_ds = torch.utils.data.random_split(ds, [int(len_ds * 0.8), int(len_ds - int(len_ds * 0.8))])
    
    # Initialize train and validation dataloaders
    train_loader = DataLoader(tr_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=8)
    
    # Function to get feature maps
    def get_fm(fm):
        
        """
        
        Gets feature map with size (bs, fm_shape, 7, 7)
        applies average pooling and returns feature map
        with shape (bs, fm_shape).
        
        Argument:
        
        fm - feature map.
        
        """
        
        pool = AvgPool2d((fm.shape[2],fm.shape[3]))
        
        return torch.reshape(pool(fm), (-1, fm.shape[1]))
    
    # Log in wandb
    os.system('wandb login 3204eaa1400fed115e40f43c7c6a5d62a0867ed1')     
    
    # Initialize a new run (a new project)
    wandb_logger = WandbLogger(name=f'{model_name}_{datetime.now().strftime("%m/%d/%H:%M:%S")}_{os.path.basename(path)}_training', project='Classification')

    print(f"Number of train set images: {len(tr_ds)}")
    print(f"Number of validation set images: {len(val_ds)}")
    print(f"\nTrain dataset has {num_classes} classes")
    
    # Initialize function to compute cosine similarity
    cos = CosineSimilarity(dim=1, eps=1e-6)
    
    # Labels for loss function
    labels = {"pos": torch.tensor(1.).unsqueeze(0),
              "neg": torch.tensor(-1.).unsqueeze(0)}
    
    assert only_features or only_labels, "Please choose at least one loss function to train the model (triplet loss or crossentropy loss)"
    if only_features and only_labels:
        print("\nTrain using triplet loss and crossentropy loss\n")
    elif only_features == True and only_labels == None:
        print("\nTrain using only triplet loss\n")                
    elif only_features == None and only_labels == True:
        print("\nTrain using only crossentropy loss\n")      

    # Model class
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
            self.cos_loss = CosineEmbeddingLoss(margin=0.2)
            self.ce_loss = CrossEntropyLoss()
            # Example input for visualizing the graph in Tensorboard
            self.example_input_array = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

        def forward(self, inp):
            return self.model(inp)
        
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
            ims, regs = batch
            
            # Get feature maps and pred labels
            lbl_ims = self.model(ims)
            loss = self.ce_loss(lbl_ims, regs)
            
            top3, top1 = 0, 0
            
            for idx, lbl_im in enumerate(lbl_ims):
                
                vals, inds = torch.topk(lbl_im, k=3)
                if regs[idx] == regs[inds[0]] or regs[idx] == regs[inds[1]] or regs[idx] == regs[inds[2]]:
                    top3 += 1
                if regs[idx] in regs[inds[0]]:
                    top1 += 1

            # Logs the loss per epoch to tensorboard (weighted average over batches)
            self.log("train_loss", loss)
            self.log("train_top3", top3 / len(lbl_ims))
            self.log("train_top1", top1 / len(lbl_ims))

            return OD([('loss', loss)]) #, ('train_top3', top3 / len(ims)), ('train_top1', top1 / len(ims))])  # Return tensor to call ".backward" on

        def validation_step(self, batch, batch_idx): # triplet loss 

            cos_sims, cos_unsims, cos_sims_pair, cos_unsims_pair = [], [], [], []
            ims, regs = batch
            
            # Get feature maps and pred labels
            lbl_ims = self.model(ims)
            loss = self.ce_loss(lbl_ims, regs)
            
            top3, top1 = 0, 0
            
            for idx, lbl_im in enumerate(lbl_ims):
                
                vals, inds = torch.topk(lbl_im, k=3)
                if regs[idx] == regs[inds[0]] or regs[idx] == regs[inds[1]] or regs[idx] == regs[inds[2]]:
                    top3 += 1
                if regs[idx] in regs[inds[0]]:
                    top1 += 1

            # Logs the loss per epoch to tensorboard (weighted average over batches)
            self.log("val_loss", loss)
            self.log("val_top3", top3 / len(lbl_ims))
            self.log("val_top1", top1 / len(lbl_ims))

            # return OD([('loss', loss), ('val_top3', top3), ('val_top1', top1)]) 
            return OD([('loss', loss), ('val_top3', top3),
                       ('cos_sims', torch.mean(torch.FloatTensor(cos_sims)))])

        def test_step(self, batch, batch_idx): # triplet loss 
            
            """ 
            Compares test images in the batch with the all images in the dataloader.
            
            """
            fms_ims, fms_poss, fms_negs, scores, cos_sims, gts_all, im_pred_lbls_all, pos_pred_lbls_all = [], [], [], [], [], [], [], []
            top1, top3, = 0, 0
            
            print("\nObtaining embeddings and predicting labels...\n")
            # Get feature maps and pred labels of the whole test data
            for i, batch_all in enumerate(test_loader):
                with torch.no_grad():
                    
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
                    gts_all.extend(regs_all.cuda())
                    im_pred_lbls_all.extend(lbl_ims_all)
                    pos_pred_lbls_all.extend(lbl_poss_all)
                
            fms_ims = torch.stack(fms_ims)
            fms_poss = torch.stack(fms_poss)
            fms_negs = torch.stack(fms_negs)
            gts_all = torch.stack(gts_all)
            im_pred_lbls_all = torch.stack(im_pred_lbls_all)
            pos_pred_lbls_all = torch.stack(pos_pred_lbls_all)
            
            print("Done!\n")               
            print("\nCalculating metrics...")
        
            # Compute loss, top3, and top1   
            for index, fm in enumerate(fms_ims):
                
                # Compute loss                
                if only_features and only_labels:
                    loss_cos_poss = self.cos_loss(fm, fms_poss, labels["pos"].to("cuda")) 
                    loss_cos_negs = self.cos_loss(fm, fms_negs, labels["neg"].to("cuda"))
                    loss_cos = loss_cos_poss + loss_cos_negs
                    loss_ce_ims = self.ce_loss(lbl_ims, regs)
                    loss_ce_poss = self.ce_loss(lbl_poss, regs)
                    loss_ce = loss_ce_ims + loss_ce_poss
                    loss = loss_cos + loss_ce
                elif only_features == True and only_labels == None:
                    loss_cos_poss = self.cos_loss(fm, fms_poss, labels["pos"].to("cuda")) 
                    loss_cos_negs = self.cos_loss(fm, fms_negs, labels["neg"].to("cuda"))
                    loss_cos = loss_cos_poss + loss_cos_negs
                    loss = loss_cos                 
                elif only_features == None and only_labels == True:
                    loss_ce_ims = self.ce_loss(lbl_ims, regs)
                    loss_ce_poss = self.ce_loss(lbl_poss, regs)
                    loss_ce = loss_ce_ims + loss_ce_poss
                    loss = loss_ce                
                
                score_pair = cos(fm.unsqueeze(0), fms_poss[index].unsqueeze(0)) # (1, fm), (len(dl), fm) = (len(dl), fm)
                scores.append(score_pair)
                cos_sim = cos(fm.unsqueeze(0), fms_poss)
                vals, inds = torch.topk(cos_sim, k=3)
                if gts_all[index] in gts_all[inds]:
                    top3 += 1
                if gts_all[index] in gts_all[inds[0]]:
                    top1 += 1                    

            print("Calculating metrics is done!\n")

            self.log("test_sim_scores", torch.mean(torch.FloatTensor(scores)).item())
            self.log("test_loss", loss / len(fms_ims))
            self.log("test_top3", top3 / len(fms_ims))
            self.log("test_top1", top1 / len(fms_ims))

    def create_model(model_name, conv_input=False, num_classes=num_classes):
        
        """ 
        
        Gets model name and creates a timm model.
        
        """

        if model_name in model_dict:
            base_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
            # base_model.head = Identity()
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
            
        return model#.to('cuda')

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
            # gpus=1 if str(device) == "cuda:1" else 0,
            precision=16, amp_backend='native',
            # total num of epochs
            max_epochs=300,
            log_every_n_steps=15,
            logger=wandb_logger,
            # auto_lr_find=True,
            # fast_dev_run=True,
            strategy="ddp", accelerator="gpu", devices=3, 
            callbacks=[
                
                ModelCheckpoint(
                    filename='{epoch}-{val_loss:.2f}-{train_loss:.2f}-{val_top1:.2f}', 
                    every_n_train_steps = None, save_top_k=1,
                    save_weights_only=True, mode="max", monitor="val_top1" 
                ),  # Save the best checkpoint based on the min val_loss recorded. Saves only weights and not optimizer
                EarlyStopping(monitor="val_top1", mode="max", patience=20, verbose=True), # set the metric (and change the mode!) to track for early stopping
                LearningRateMonitor("epoch"), # Log learning rate every epoch
            ]
        )
        trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filename = os.path.join(sp, 'rexnet_150_Adam_0.0003', 'Image Retrieval', "1tgu7vtc", "checkpoints")
        # pretrained_filename = pretrained_filename + '/epoch=3-val_loss=4.73-cos_sims=0.91-val_top1=0.48.ckpt'
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

        return model

    
    trained_model = train_model(
    model_name=model_name, optimizer_name=optimizer_name, save_name=f"{model_name}_{optimizer_name}_{lr}",
    optimizer_hparams=optimizer_hparams)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Triplet Loss PyTorch Lightning Arguments')
    parser.add_argument('-ed', '--expdir', default=None, help='Experiment directory')
    parser.add_argument("-sp", "--save_path", type=str, default='saved_models', help="Path to save trained models")
#     parser.add_argument('-cp', '--checkpoint_path', type=str, default="/home/ubuntu/workspace/bekhzod/triplet-loss-pytorch/pytorch_lightning/saved_models/model_best.pth.tar", help='Path to the trained model')
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-d", "--device", type=str, default='cuda:1', help="GPU device number")
    parser.add_argument("-ip", "--ims_path", type=str, default='/home/ubuntu/workspace/bekhzod/pytorch-image-models/dataset/real', help="Path to the images")
    parser.add_argument("-mn", "--model_name", type=str, default='swin_s3_base_224', help="Model name (from timm library (ex. darknet53, ig_resnext101_32x32d))")
    parser.add_argument("-on", "--optimizer_name", type=str, default='Adam', help="Optimizer name (Adam or SGD)")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate value")
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-5, help="Weight decay value")
    parser.add_argument("-ofm", "--only_feature_embeddings", type=bool, default=None,)
    parser.add_argument("-otl", "--only_target_labels", type=bool, default=True,
                        help="If True trains the model using only cross entropy and and return predicted labels (if both otl and ofm are True uses two loss functions simultaneously)")
    
    args = parser.parse_args() 
    
    run(args) 
