# Import libraries
import os, argparse, yaml, torch, torchvision, timm, pickle, wandb, AutoAugment
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
from contrastive_loss import ContrastiveLoss
from sketch_dataset import SketchyImageDataset
from tqdm import tqdm
import torch.nn.functional as F

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
    
    # Initialize train arguments
    model_dict = {}
    optimizer_hparams={"lr": lr, "weight_decay": wd}
    model_dict[model_name] = 0 
    
    # Login to wandb: change to your account with the corresponding key
    os.system('wandb login 3204eaa1400fed115e40f43c7c6a5d62a0867ed1')     
    
    
    # Get train transformations
    transformations = {}   
    transformations['qry'] = transforms.Compose([
                            transforms.Resize((224,224)),
                            AutoAugment.ImageNetPolicy(),
                            transforms.ToTensor(),
                                                  ])
    transformations['pos'] = transforms.Compose([
        transforms.Resize((224,224)),
        AutoAugment.ImageNetPolicy(),
        transforms.ToTensor(),
    ])  
    transformations['neg'] = transforms.Compose([
        transforms.Resize((224,224)),
        AutoAugment.ImageNetPolicy(),
        transforms.ToTensor(),
    ])

    # Set the path to the json file with train, val, test split
    out_path = "data/sketchy_database_256_soft_split_cat.json"

    # Get train, validation, and test datasets
    tr_ds = SketchyImageDataset(data_dir = path, transform_dic=transformations, random=True, trainval_json=out_path, trainval='train', load_images=False)
    val_ds = SketchyImageDataset(data_dir = path, transform_dic=transformations, random=True, trainval_json=out_path, trainval='val', load_images=False)
    test_ds = SketchyImageDataset(data_dir = path, transform_dic=transformations, random=True, trainval_json=out_path, trainval='test', load_images=False)
    
    # Initialize project in wandb    
    wandb_logger = WandbLogger(name=f'{model_name}_{datetime.now().strftime("%m/%d/%H:%M:%S")}_{bs}_{lr}', 
                               project='Sketchy-Dataset-Training')
    
    # Get number of classes
    num_classes = tr_ds.get_cat_length()
    
    # Double check with prints
    print(f"Number of train set images: {len(tr_ds)}")
    print(f"Number of validation set images: {len(val_ds)}")
    print(f"Number of test set images: {len(test_ds)}")
    print(f"\nTrain dataset has {num_classes} classes")
    print(f"Validation dataset has {val_ds.get_cat_length()} classes")    
    print(f"Test dataset has {test_ds.get_cat_length()} classes")
    
    # Initialize cosine similarity computation function
    cos = CosineSimilarity(dim = 1, eps = 1e-6)
    
    # Initialize train, validation, and test dataloaders
    train_loader = DataLoader(tr_ds, batch_size = bs, shuffle = True, drop_last = True, num_workers = 8)
    val_loader = DataLoader(val_ds, batch_size = bs, shuffle = True, drop_last = True, num_workers = 8)
    test_loader = DataLoader(test_ds, batch_size = bs, shuffle = True, drop_last = True, num_workers = 8)  
    
    # Initialize labels for the loss functions
    labels = {"con_pos": torch.tensor(1.).unsqueeze(0),
              "con_neg": torch.tensor(0.).unsqueeze(0),
              "cos_pos": torch.tensor(1.).unsqueeze(0),
              "cos_neg": torch.tensor(-1.).unsqueeze(0)}
    
    # Function to get feature maps
    def get_fm(fm):
        
        """
        
        This function gets feature map with size (bs, fm_shape, 7, 7)
        applies average pooling and returns feature map with shape (bs, fm_shape).
        
        Parameter:
        
            fm - feature map, tensor.
        
        Output:
        
            fm - reshaped feature map, tensor.
        
        """
        
        pool = AvgPool2d((fm.shape[2],fm.shape[3]))
        
        return torch.reshape(pool(fm), (-1, fm.shape[1]))
    
    assert only_features or only_labels, "Please choose at least one loss function to train the model (triplet loss or crossentropy loss)"
    if only_features and only_labels: print("\nTrain using triplet loss and crossentropy loss\n")
    elif only_features == True and only_labels == None: print("\nTrain using only triplet loss\n")                
    elif only_features == None and only_labels == True: print("\nTrain using only crossentropy loss\n")      
    
    # Model class
    class Model(pl.LightningModule):

        def __init__(self, model_name,  optimizer_name, optimizer_hparams):
            
            """
            
            This class gets model name, optimizer name and hparams and returns trained model (pytorch lightning) with results (dict).
            
            Parameters:
            
                model_name          - model name in the timm library, str;
                optimizer_name      - optimizer name in the torch library, str;
                optimizer_hparams   - hyperparameters of the optimizer, list.
            
            Output:
            
                train process.
            
            """
            
            super().__init__()
            # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
            self.save_hyperparameters()

            # Create model
            self.model = create_model(model_name)
            
            # Create loss modules
            self.cos_loss = CosineEmbeddingLoss(margin = 0.3)
            self.con_loss = ContrastiveLoss(margin = 0.3)
            self.ce_loss = CrossEntropyLoss()
            
            # Initialize a sample array
            self.example_input_array = torch.zeros((1, 3, 224, 224), dtype = torch.float32)

        def forward(self, inp): return self.model(inp)
        
        def configure_optimizers(self):
            
            """
            
            This function initializes optimizer and scheduler.
            
            Outputs:
            
                optimizer - optimizer to update trainable parameters of the model;
                scheduler - scheduler of the optimizer.
            
            """
            
            assert self.hparams.optimizer_name in ["Adam", "SGD"], "Currently only Adam and SGD optimizer are supported."
            if self.hparams.optimizer_name == "Adam":  optimizer = torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
            elif self.hparams.optimizer_name == "SGD": optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
            
            # Set the scheduler
            scheduler = MultiStepLR(optimizer = optimizer, milestones = [6,15,22,30,35,40], gamma = 0.1, verbose = True)
        
            return [optimizer], [scheduler]
        
        # One training step
        def training_step(self, batch, batch_idx): 
            
            """
            
            This function gets batch and batch index and does one train step and returns train loss.
            
            Parameters:
            
                batch     - one batch with images, tensor;
                batch_idx - index of the batch, int.
                
            Output:
            
                loss      - loss of training model for the current batch, tensor.
            
            """
            
            # Initialize a list to track cosine similarities
            cos_sims = []
            ims, poss, negs, clss, regs = batch['qry'], batch['pos'][0], batch['neg'][0], batch['cat_idx'], batch['prod_idx']
            
            # Get feature maps and pred labels of query images
            fm_ims = self.model.forward_features(ims)
            fm_ims = get_fm(fm_ims)
            lbl_ims = self.model.classifier(fm_ims) 
            
            # Get feature maps and pred labels of positive images
            fm_poss = self.model.forward_features(poss)
            fm_poss = get_fm(fm_poss)
            lbl_poss = self.model.classifier(fm_poss)
            
            # Get feature maps and pred labels of negative images
            fm_negs = self.model.forward_features(negs)
            fm_negs = get_fm(fm_negs)
            lbl_negs = self.model.classifier(fm_negs)
            
            # Compute loss
            if only_features and only_labels:
                
                # Cosine embedding loss
                loss_cos_poss = self.cos_loss(fm_ims, fm_poss, labels["cos_pos"].to("cuda")) 
                loss_cos_negs = self.cos_loss(fm_ims, fm_negs, labels["cos_neg"].to("cuda"))
                loss_cos = loss_cos_poss + loss_cos_negs
                
                # Contrastive loss
                loss_con_poss = self.con_loss(fm_ims, fm_poss, labels["con_pos"].to("cuda")) 
                loss_con_negs = self.con_loss(fm_ims, fm_negs, labels["con_neg"].to("cuda"))
                loss_con = loss_con_poss + loss_con_negs
                
                # Cross entropy loss
                loss_ce_ims = self.ce_loss(lbl_ims, clss)
                loss_ce_poss = self.ce_loss(lbl_poss, clss)
                loss_ce = loss_ce_ims + loss_ce_poss
                
                # Total loss
                loss = loss_cos + loss_con + loss_ce
                
            elif only_features == True and only_labels == None:
                
                # Cosine embedding loss
                loss_cos_poss = self.cos_loss(fm_ims, fm_poss, labels["pos"].to("cuda")) 
                loss_cos_negs = self.cos_loss(fm_ims, fm_negs, labels["neg"].to("cuda"))
                loss_cos = loss_cos_poss + loss_cos_negs
                
                # Total loss
                loss = loss_cos    
                
            elif only_features == None and only_labels == True:
                
                # Cross entropy loss
                loss_ce_ims = self.ce_loss(lbl_ims, regs)
                loss_ce = loss_ce_ims
                
                # Total loss
                loss = loss_ce
                
            # Initialize top3 and top1
            top3, top1 = 0, 0            
            
            # Go through every predicted label and feature map
            for idx, lbl_im in (enumerate(lbl_ims)):
                
                # Get feature map similarities
                sim = cos(fm_ims[idx].unsqueeze(0), fm_poss)  # batch                
                
                # Get top3 values and indices of the predictions
                vals, inds = torch.topk(sim, k = 3)
                
                # Compute top3
                if clss[idx] == clss[inds[0]] or clss[idx] == clss[inds[1]] or clss[idx] == clss[inds[2]]: top3 += 1
                # Compute top1
                if clss[idx] in clss[inds[0]]: top1 += 1
                    

            # Logs the loss per epoch to tensorboard (weighted average over batches)
            self.log("train_loss", loss)
            self.log("train_top3", top3 / len(lbl_ims))
            self.log("train_top1", top1 / len(lbl_ims))

            return OD([('loss', loss)]) 

        # Validation step of the training process
        def validation_step(self, batch, batch_idx): 
            
            """
            
            This function gets batch and batch index and does one validation step and returns validation loss.
            
            Parameters:
            
                batch           - one batch with images, tensor;
                batch_idx       - index of the batch, int.
                
            Outputs:
            
                val_loss        - loss of validation process for the current batch, tensor.
                val_top3        - top3 validation accuracy of the training model for the current batch, tensor.
                cos_sims        - cosine similarities of the current batch, list.
            
            """
            
            # Initialize lists to track validation process results
            cos_sims, cos_unsims, cos_sims_pair, cos_unsims_pair = [], [], [], []
            ims, poss, negs, clss, regs = batch['qry'], batch['pos'][0], batch['neg'][0], batch['cat_idx'], batch['prod_idx']

             # Get feature maps and pred labels of query images
            fm_ims = self.model.forward_features(ims)
            fm_ims = get_fm(fm_ims)
            lbl_ims = self.model.classifier(fm_ims) 
            
            # Get feature maps and pred labels of positive images
            fm_poss = self.model.forward_features(poss)
            fm_poss = get_fm(fm_poss)
            lbl_poss = self.model.classifier(fm_poss)
            
            # Get feature maps and pred labels of negative images
            fm_negs = self.model.forward_features(negs)
            fm_negs = get_fm(fm_negs)
            lbl_negs = self.model.classifier(fm_negs)
            
            # Compute loss
            if only_features and only_labels:
                
                # Cosine embedding loss
                loss_cos_poss = self.cos_loss(fm_ims, fm_poss, labels["cos_pos"].to("cuda")) 
                loss_cos_negs = self.cos_loss(fm_ims, fm_negs, labels["cos_neg"].to("cuda"))
                loss_cos = loss_cos_poss + loss_cos_negs
                
                # Contrastive loss
                loss_con_poss = self.con_loss(fm_ims, fm_poss, labels["con_pos"].to("cuda")) 
                loss_con_negs = self.con_loss(fm_ims, fm_negs, labels["con_neg"].to("cuda"))
                loss_con = loss_con_poss + loss_con_negs
                
                # Cross entropy loss
                loss_ce_ims = self.ce_loss(lbl_ims, clss)
                loss_ce_poss = self.ce_loss(lbl_poss, clss)
                loss_ce = loss_ce_ims + loss_ce_poss
                
                # Total loss
                loss = loss_cos + loss_con + loss_ce
                
            elif only_features == True and only_labels == None:
                
                # Cosine embedding loss
                loss_cos_poss = self.cos_loss(fm_ims, fm_poss, labels["pos"].to("cuda")) 
                loss_cos_negs = self.cos_loss(fm_ims, fm_negs, labels["neg"].to("cuda"))
                loss_cos = loss_cos_poss + loss_cos_negs
                
                # Total loss
                loss = loss_cos    
                
            elif only_features == None and only_labels == True:
                
                # Cross entropy loss
                loss_ce_ims = self.ce_loss(lbl_ims, regs)
                loss_ce = loss_ce_ims
                
                # Total loss
                loss = loss_ce
                
            # Compute top3 and top1
            top3, top1 = 0, 0
            
            # Go through predicted labels and feature maps
            for idx, lbl_im in (enumerate(lbl_ims)):
                
                # Compute cosine similarities of a query image with the corresponding positive image
                sim_pair = cos(fm_ims[idx].unsqueeze(0), fm_poss[idx].unsqueeze(0)) 
                cos_sims_pair.append(sim_pair)
                
                # Compute cosine similarities of a query image with the corresponding negative image
                unsim_pair = cos(fm_ims[idx].unsqueeze(0), fm_negs[idx].unsqueeze(0)) 
                cos_unsims_pair.append(unsim_pair)
                
                # Compute cosine similarities of a query image with all the images in the dataloader
                sim = cos(fm_ims[idx].unsqueeze(0), fm_poss)  # batch                
                
                # Get top3                
                vals, inds = torch.topk(sim, k=3)
                # Compute top3
                if clss[idx] == clss[inds[0]] or clss[idx] == clss[inds[1]] or clss[idx] == clss[inds[2]]: top3 += 1
                # Compute top1
                if clss[idx] in clss[inds[0]]: top1 += 1

            # Logs the loss per epoch to tensorboard (weighted average over batches)
            self.log("val_loss", loss)
            self.log("val_loss_cos_poss", loss_cos_poss)
            self.log("val_loss_cos_negs", loss_cos_negs)
            self.log("val_loss_con_poss", loss_con_poss)
            self.log("val_loss_con_negs", loss_con_negs)
            self.log("val_loss_ce_ims", loss_ce_ims)
            self.log("val_loss_ce_poss", loss_ce_poss)
            self.log("cos_sims", torch.mean(torch.FloatTensor(cos_sims_pair)).item())
            self.log("cos_unsims", torch.mean(torch.FloatTensor(cos_unsims_pair)).item())
            self.log("val_top3", top3 / len(lbl_ims))
            self.log("val_top1", top1 / len(lbl_ims))
            
            return OD([('loss', loss), ('val_top3', top3), ('cos_sims', torch.mean(torch.FloatTensor(cos_sims)))])

    def create_model(model_name, conv_input = False, num_classes = num_classes):
        
        """ 
        
        This function gets model name and creates a timm model.
        
        Parameters:
        
            model_name  - name of the model in timm library, str;
            conv_input  - option for the the input to pass convolution layer first, bool;
            num_classes - number of classes in the dataset.
            
        Output:
        
            model       - created model from timm library.
        
        """

        # Find a model name in the dictionary
        if model_name in model_dict:
            base_model = timm.create_model(model_name, pretrained = True, num_classes = num_classes)
            print(f"Model {model_name} with the best weights is successfully loaded!")  
            model = base_model
        
        # Model name is not in the dictionary
        else:
            assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'
            
        return model

    # Train function
    def train_model(model_name, save_name=None, **kwargs):
        
        """
        
        This function trains the model and returns trained model with its results.
        
        Parameters:
        
            model_name - name of the model in timm library, str;
            save_name  - name for the saved model, str.
            
        Outputs:
        
            model      - a trained model, timm model.
            
        """
        
        # Set the save name
        if save_name is None:
            save_name = model_name

        # Create a PyTorch Lightning trainer with callbacks
        trainer = pl.Trainer(
            default_root_dir=os.path.join(sp, save_name),  # dir name to save models
            # amp
            precision = 16, amp_backend = 'native',
            # total num of epochs
            max_epochs = 300,
            # logger
            logger = wandb_logger,
            # log_steps
            log_every_n_steps = 15,
            # parallel computing
            strategy = "ddp", accelerator = "gpu", devices = 3, 
            # callbacks
            callbacks=[
                ModelCheckpoint(
                    # name to save the model
                    filename='{epoch}-{val_loss:.2f}-{cos_sims:.2f}-{val_top1:.2f}', 
                    every_n_train_steps = None, save_top_k = 1,
                    # metrics to track
                    save_weights_only = True, mode = "max", monitor = "cos_sims" 
                ),
                
                EarlyStopping(
                    # metric to track
                    monitor = "cos_sims", mode = "max",
                    # number of epochs to wait
                    patience = 10, verbose = True
                ), 
                
                # Log learning rate every epoch
                LearningRateMonitor("epoch") 
            ]
        )
        
        # Train loggers
        # Plot computation graph in TB
        trainer.logger._log_graph = True  
        # Optional logging argument
        trainer.logger._default_hp_metric = None  

        # Check the availability of the pretrained file
        pretrained_filename = os.path.join(sp, 'efficientnet_b3a_Adam_0.0047863', 'Sketchy-Dataset-Training', "3qz9sxmq", "checkpoints")
        # pretrained_filename = pretrained_filename + '/epoch=22-val_loss=0.31-cos_sims=0.93-val_top1=0.96.ckpt'
        if os.path.isfile(pretrained_filename):
            print(f"Found pretrained model at {pretrained_filename}, loading...")
            # Load model from the checkpoint
            model = Model.load_from_checkpoint(pretrained_filename)
        else:
            pl.seed_everything(42)
            # Initialize a new model 
            model = Model(model_name = model_name, **kwargs)
            # Fit the dataloaders to the model            
            trainer.fit(model, train_loader, val_loader)

        return model
    
    # Training process
    trained_model = train_model(model_name = model_name,
                                optimizer_name = optimizer_name,
                                save_name = f"{model_name}_{optimizer_name}_{lr}",
                                optimizer_hparams = optimizer_hparams)

if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description='Sketchy Database Triplet Train PyTorch Lightning Arguments')
    
    # Add arguments to the parser
    parser.add_argument('-ed', '--expdir', default=None, help='Experiment directory')
    parser.add_argument("-sp", "--save_path", type=str, default='saved_models', help="Path to save trained models")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-d", "--device", type=str, default='cuda:1', help="GPU device number")
    parser.add_argument("-ip", "--ims_path", type=str, default='/home/ubuntu/workspace/dataset/sketchy_database_256', help="Path to the images")
    parser.add_argument("-mn", "--model_name", type=str, default='efficientnet_b3a', help="Model name (from timm library (ex. darknet53, ig_resnext101_32x32d))")
    parser.add_argument("-on", "--optimizer_name", type=str, default='Adam', help="Optimizer name (Adam or SGD)")
    parser.add_argument("-lr", "--learning_rate", type=float, default=4.7863e-03, help="Learning rate value") # from find_lr
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-5, help="Weight decay value")
    parser.add_argument("-ofm", "--only_feature_embeddings", type=bool, default=True,
                        help="If True trains the model using only triplet loss and and return feature embeddings (if both otl and ofm are True uses two loss functions simultaneously)")
    parser.add_argument("-otl", "--only_target_labels", type=bool, default=True,
                        help="If True trains the model using only cross entropy and and return predicted labels (if both otl and ofm are True uses two loss functions simultaneously)")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the script
    run(args)
