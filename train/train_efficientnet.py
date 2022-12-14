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
from sketch_dataset import SketchyImageDataset
from tqdm import tqdm

def run(args):
    
    # Get the training arguments
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

    # Set train dictionaries
    model_dict = {}
    
    # Optimizer
    optimizer_hparams={"lr": lr, "weight_decay": wd}
    
    # Model
    model_dict[model_name] = 0 
    
    # Augmentations
    transformations = {}   

    # Query Image Transformations
    transformations['qry'] = transforms.Compose([
                             transforms.Resize((224,224)),
                             AutoAugment.ImageNetPolicy(),
                             transforms.ToTensor()])
    
    # Positive Image Transformations
    transformations['pos'] = transforms.Compose([
        transforms.Resize((224,224)),
        AutoAugment.ImageNetPolicy(),
        transforms.ToTensor()])  
    
    # Negative Image Transformations
    transformations['neg'] = transforms.Compose([
        transforms.Resize((224,224)),
        AutoAugment.ImageNetPolicy(),
        transforms.ToTensor()])
    
    # Wandb Login 
    os.system('wandb login 3204eaa1400fed115e40f43c7c6a5d62a0867ed1')     
    
    # Set path to the json file with data split
    out_path = "data/sketchy_database_256_soft_split_cat.json"

    # Get train, validation, and test datasets
    tr_ds = SketchyImageDataset(data_dir = path, transform_dic=transformations, random=True, trainval_json=out_path, trainval='train', load_images=False)
    val_ds = SketchyImageDataset(data_dir = path, transform_dic=transformations, random=True, trainval_json=out_path, trainval='val', load_images=False)
    test_ds = SketchyImageDataset(data_dir = path, transform_dic=transformations, random=True, trainval_json=out_path, trainval='test', load_images=False)
    
    # Initialize project in wandb
    wandb_logger = WandbLogger(name=f'{model_name}_{datetime.now().strftime("%m/%d/%H:%M:%S")}_{bs}_{lr}', project='Sketchy-Dataset-Training')

    # Get number of classes
    num_classes = tr_ds.get_cat_length()
    print(f"Number of train set images: {len(tr_ds)}")
    print(f"Number of validation set images: {len(val_ds)}")
    print(f"Number of test set images: {len(test_ds)}")
    print(f"\nTrain dataset has {num_classes} classes")
    print(f"Validation dataset has {val_ds.get_cat_length()} classes")    
    print(f"Test dataset has {test_ds.get_cat_length()} classes")
    
    # Initialize function to compute cosine similarity
    cos = CosineSimilarity(dim=1, eps=1e-6)
    
    # Initialize train, validation, and test dataloaders
    train_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=8)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=8)  
    
    # Set labels for loss functions
    labels = {"pos": torch.tensor(1.).unsqueeze(0),
              "neg": torch.tensor(-1.).unsqueeze(0)}

    # Function to get feature maps
    def get_fm(fm):
        
        """
        
        Gets feature maps with size (bs, fm, 7, 7), applies average pooling
        and returns feature maps with (bs, fm) shape.
        
        Argument:
        fm - feature map
        
        """
        
        # Initialize average pooling layer
        pool = AvgPool2d((fm.shape[2],fm.shape[3]))
        
        return torch.reshape(pool(fm), (-1, fm.shape[1]))
    
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
            
            # Create loss modules
            self.cos_loss = CosineEmbeddingLoss(margin=0.5)
            self.ce_loss = CrossEntropyLoss()
            
            # Initialize example array
            self.example_input_array = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

        def forward(self, inp):
            return self.model(inp)
        
        # Optimizers function
        def configure_optimizers(self):
            
            """
            
            Gets optimizer name initialize it and returns the optimizer with the scheduler.
            
            """
            
            # AdamW optimizer
            if self.hparams.optimizer_name == "Adam":
                # AdamW is Adam with a correct implementation of weight decay (see here
                # for details: https://arxiv.org/pdf/1711.05101.pdf)
                optimizer = torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
                
            # SGD optimizer
            elif self.hparams.optimizer_name == "SGD":
                optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
            
            # Other optimizers
            else:
                assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'
            
            # Scheduler for the optimizers
            scheduler = MultiStepLR(optimizer=optimizer, milestones=[6,12,20,30,35,40], gamma=0.1, verbose=True)
        
            return [optimizer], [scheduler]
        
        # Triplet model training step
        def training_step(self, batch, batch_idx):
            
            """ 
            
            Gets batch and batch index performs one step of training and returns train loss. 
            
            Arguments:
            batch - one batch of the train dataloader;
            batch_idx - batch index of the corresponding batch from the train dataloader.
            
            """
            
            # Initialize list to track cosine similarities
            cos_sims = []
            
            # Get images and labels
            ims, poss, negs, clss, regs = batch['qry'], batch['pos'][0], batch['neg'][0], batch['cat_idx'], batch['prod_idx']
            
            # Get feature maps of query images
            fm_ims = self.model.forward_features(ims)
            fm_ims = get_fm(fm_ims)
            
            # Get predicted labels for the query images
            lbl_ims = self.model.classifier(fm_ims) 
            
            # Get feature maps of positive images
            fm_poss = self.model.forward_features(poss)
            fm_poss = get_fm(fm_poss)
            
            # Get predicted labels for the positive images
            lbl_poss = self.model.classifier(fm_poss)
            
            # Get feature maps of negative images
            fm_negs = self.model.forward_features(negs)
            fm_negs = get_fm(fm_negs)
            
            # Get predicted labels for the negative images
            lbl_negs = self.model.classifier(fm_negs)
            
            # Compute loss
            if only_features and only_labels:
                
                # Cosine Embedding Loss
                loss_cos_poss = self.cos_loss(fm_ims, fm_poss, labels["pos"].to("cuda")) 
                loss_cos_negs = self.cos_loss(fm_ims, fm_negs, labels["neg"].to("cuda"))
                loss_cos = loss_cos_poss + loss_cos_negs
                
                # Cross Entropy Loss
                loss_ce_ims = self.ce_loss(lbl_ims, clss)
                loss_ce_poss = self.ce_loss(lbl_poss, clss)
                loss_ce = loss_ce_ims + loss_ce_poss
                loss = loss_cos + loss_ce
                
            elif only_features == True and only_labels == None:
                
                # Cosine Embedding Loss
                loss_cos_poss = self.cos_loss(fm_ims, fm_poss, labels["pos"].to("cuda")) 
                loss_cos_negs = self.cos_loss(fm_ims, fm_negs, labels["neg"].to("cuda"))
                loss_cos = loss_cos_poss + loss_cos_negs
                loss = loss_cos   
                
            elif only_features == None and only_labels == True:
                
                # Cross Entropy Loss
                loss_ce_ims = self.ce_loss(lbl_ims, regs)
                loss_ce = loss_ce_ims
                loss = loss_ce
                
            # Set initial top3, top1
            top3, top1 = 0, 0       
            
            # Go through every predicted label
            for idx, lbl_im in (enumerate(lbl_ims)):
                
                # Compute cosine similarity of the feature map with the all feature maps in the batch
                sim = cos(fm_ims[idx].unsqueeze(0), fm_poss)
                
                # Get top3 values and indices
                vals, inds = torch.topk(sim, k=3)
                
                # Compute top3
                if clss[idx] == clss[inds[0]] or clss[idx] == clss[inds[1]] or clss[idx] == clss[inds[2]]: top3 += 1
                
                # Compute top1
                if clss[idx] in clss[inds[0]]: top1 += 1

            # Wandb logs
            self.log("train_loss", loss)
            self.log("train_top3", top3 / len(lbl_ims))
            self.log("train_top1", top1 / len(lbl_ims))

            return OD([('loss', loss)]) 

        # Function for validation step
        def validation_step(self, batch, batch_idx): 
            
            """ 
            
            Gets batch and batch index performs one step of validation and returns validation loss along with cosine similarities. 
            
            Arguments:
            batch - one batch of the validation dataloader;
            batch_idx - batch index of the corresponding batch from the validation dataloader.
            
            """
            
            # Switch to evaluation mode
            self.model.eval()
            
            # Initialize lists to track validation progress
            cos_sims, cos_unsims, cos_sims_pair, cos_unsims_pair = [], [], [], []
            
            # Get images and labels
            ims, poss, negs, clss, regs = batch['qry'], batch['pos'][0], batch['neg'][0], batch['cat_idx'], batch['prod_idx']
            
            # Get feature maps of query images
            fm_ims = self.model.forward_features(ims)
            fm_ims = get_fm(fm_ims)
            
            # Get predicted labels for the query images
            lbl_ims = self.model.classifier(fm_ims) 
            
            # Get feature maps of positive images
            fm_poss = self.model.forward_features(poss)
            fm_poss = get_fm(fm_poss)
            
            # Get predicted labels for the positive images
            lbl_poss = self.model.classifier(fm_poss)
            
            # Get feature maps of negative images
            fm_negs = self.model.forward_features(negs)
            fm_negs = get_fm(fm_negs)
            
            # Get predicted labels for the negative images
            lbl_negs = self.model.classifier(fm_negs)          
            
            # Compute loss
            if only_features and only_labels:
                
                # Cosine Embedding Loss
                loss_cos_poss = self.cos_loss(fm_ims, fm_poss, labels["pos"].to("cuda")) 
                loss_cos_negs = self.cos_loss(fm_ims, fm_negs, labels["neg"].to("cuda"))
                loss_cos = loss_cos_poss + loss_cos_negs
                
                # Cross Entropy Loss
                loss_ce_ims = self.ce_loss(lbl_ims, clss)
                loss_ce_poss = self.ce_loss(lbl_poss, clss)
                loss_ce = loss_ce_ims + loss_ce_poss
                loss = loss_cos + loss_ce
                
            elif only_features == True and only_labels == None:
                
                # Cosine Embedding Loss
                loss_cos_poss = self.cos_loss(fm_ims, fm_poss, labels["pos"].to("cuda")) 
                loss_cos_negs = self.cos_loss(fm_ims, fm_negs, labels["neg"].to("cuda"))
                loss_cos = loss_cos_poss + loss_cos_negs
                loss = loss_cos   
                
            elif only_features == None and only_labels == True:
                
                # Cross Entropy Loss
                loss_ce_ims = self.ce_loss(lbl_ims, regs)
                loss_ce = loss_ce_ims
                loss = loss_ce
                
            # Set initial top3 and top1
            top3, top1 = 0, 0

            # Go through every single predicted label
            for idx, lbl_im in (enumerate(lbl_ims)):
                
                # Compute cosine similarity of the query image with the corresponding positive image
                sim_pair = cos(fm_ims[idx].unsqueeze(0), fm_poss[idx].unsqueeze(0)) 
                
                # Compute cosine similarity of the query image with the corresponding negative image
                unsim_pair = cos(fm_ims[idx].unsqueeze(0), fm_negs[idx].unsqueeze(0)) 
                
                # Compute cosine similarities of the query image with all feature maps in the dataloader
                sim = cos(fm_ims[idx].unsqueeze(0), fm_poss)                
                cos_sims_pair.append(sim_pair)
                cos_unsims_pair.append(unsim_pair)
                # vals, inds = torch.topk(lbl_im, k=3)
                vals, inds = torch.topk(sim, k=3)
                if clss[idx] == clss[inds[0]] or clss[idx] == clss[inds[1]] or clss[idx] == clss[inds[2]]:
                    top3 += 1
                if clss[idx] in clss[inds[0]]:
                    top1 += 1

            # Logs the loss per epoch to tensorboard (weighted average over batches)
            self.log("val_loss", loss)
            self.log("val_loss_cos_poss", loss_cos_poss)
            self.log("val_loss_cos_negs", loss_cos_negs)
            self.log("val_loss_ce_ims", loss_ce_ims)
            self.log("val_loss_ce_poss", loss_ce_poss)
            self.log("cos_sims", torch.mean(torch.FloatTensor(cos_sims_pair)).item())
            self.log("cos_unsims", torch.mean(torch.FloatTensor(cos_unsims_pair)).item())
            self.log("val_top3", top3 / len(lbl_ims))
            self.log("val_top1", top1 / len(lbl_ims))
            self.model.train()
            
            return OD([('loss', loss), ('cos_sims', torch.mean(torch.FloatTensor(cos_sims)))])

    def create_model(model_name, conv_input=False, num_classes=num_classes):
        
        """ 
        
        Gets model name and creates a timm model.
        
        """

        # Check the model name in the model dictionary
        if model_name in model_dict:
            base_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
            print(f"Model {model_name} with the best weights is successfully loaded!")  
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
        
        # Set the save name
        if save_name is None:
            save_name = model_name

        # Create a PyTorch Lightning trainer with the generation callback
        trainer = pl.Trainer(
            
            # path to save the model checkpoint
            default_root_dir = os.path.join(sp, save_name),  
            # set amp option
            precision = 16, amp_backend = 'native',
            # epochs to train the model
            max_epochs = 300,
            # log steps            
            log_every_n_steps = 15,
            # logger
            logger = wandb_logger,
            # parallel computing options
            strategy = "ddp", accelerator = "gpu", devices = 3, 
            # callbacks
            callbacks=[   
                # Save model checkpoint
                ModelCheckpoint(
                    # filename to save the checkpoint
                    filename='{epoch}-{val_loss:.2f}-{cos_sims:.2f}-{val_top1:.2f}', 
                    # saving options
                    every_n_train_steps = None, save_top_k = 1,
                    # metric tracking options
                    save_weights_only = True, mode = "max", monitor = "cos_sims" 
                ), 
                # Early stopping checkpoint
                EarlyStopping(
                    # metric tracking options
                    monitor = "cos_sims", mode = "max", patience = 10, verbose = True),
                
                # Log learning rate every epoch
                LearningRateMonitor("epoch"), 
            ]
        )
        
        # Logger options
        trainer.logger._log_graph = True  
        trainer.logger._default_hp_metric = None

        # Pretrained file check
        pretrained_filename = os.path.join(sp, 'rexnet_150_Adam_0.0003', 'Image Retrieval', "1tgu7vtc", "checkpoints")
        # pretrained_filename = pretrained_filename + '/epoch=3-val_loss=4.73-cos_sims=0.91-val_top1=0.48.ckpt'
        
        # If checkpoint exists
        if os.path.isfile(pretrained_filename):
            print(f"Found pretrained model at {pretrained_filename}, loading...")
            # Load the pretrained model from the checkpoint
            model = Model.load_from_checkpoint(pretrained_filename)
        
        # No checkpoint 
        else:
            pl.seed_everything(42) 
            
            # Create a model
            model = Model(model_name = model_name, **kwargs)
            
            # Fit train and validation dataloaders
            trainer.fit(model, train_loader, val_loader)

        return model
    
    # Start training
    trained_model = train_model(
    model_name=model_name, optimizer_name=optimizer_name, save_name=f"{model_name}_{optimizer_name}_{lr}",
    optimizer_hparams=optimizer_hparams)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Triplet Loss PyTorch Lightning Arguments')
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
    
    args = parser.parse_args() 
    
    run(args) 
