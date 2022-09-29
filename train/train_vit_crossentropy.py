# Import libraries
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
    model_name = args.model_name
    optimizer_name = args.optimizer_name
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
    train_loader = DataLoader(tr_ds, batch_size = bs, shuffle = False, drop_last = False, num_workers = 8)
    val_loader = DataLoader(val_ds, batch_size = bs, shuffle = False, drop_last = False, num_workers = 8)
    
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

            # Create a model
            self.model = create_model(model_name)
            # Create loss modules
            self.cos_loss, self.ce_loss = CosineEmbeddingLoss(margin = 0.2), CrossEntropyLoss()
            
            # Example input
            self.example_input_array = torch.zeros((1, 3, 224, 224), dtype = torch.float32)

        def forward(self, inp): return self.model(inp)
        
        # Optimizer configurations
        def configure_optimizers(self):
            
            """
            
            This function initializes optimizer and scheduler.
            
            Outputs:
            
                optimizer - optimizer to update trainable parameters of the model;
                scheduler - scheduler of the optimizer.
            
            """
            
            # AdamW
            if self.hparams.optimizer_name == "Adam":
                # AdamW is Adam with a correct implementation of weight decay (see here
                # for details: https://arxiv.org/pdf/1711.05101.pdf)
                optimizer = torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
            
            # SGD
            elif self.hparams.optimizer_name == "SGD":
                optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
            
            # Other cases
            else:
                assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'
            
            # Initialize scheduler
            scheduler = MultiStepLR(optimizer = optimizer, milestones = [10, 20, 30, 40, 50], gamma = 0.1, verbose = True)
        
            return [optimizer], [scheduler]
        
        def training_step(self, batch, batch_idx): 
            
            """
            
            This function gets batch and batch index and does one train step and returns train loss.
            
            Parameters:
            
                batch - one batch with images;
                batch_idx - index of the batch.
                
            """
            
            # Initialize list to track cosine similarities 
            cos_sims = []
            
            # Get images and labels
            ims, regs = batch
            
            # Get feature maps and pred labels
            lbl_ims = self.model(ims)
            
            # Compute loss
            loss = self.ce_loss(lbl_ims, regs)
            
            # Set initial top3, top1 
            top3, top1 = 0, 0
            
            # Go through every predicted label
            for idx, lbl_im in enumerate(lbl_ims):
                
                # Get top3 values and indices
                vals, inds = torch.topk(lbl_im, k=3)
                
                # Top3
                if regs[idx] == regs[inds[0]] or regs[idx] == regs[inds[1]] or regs[idx] == regs[inds[2]]: top3 += 1
                
                # Top1
                if regs[idx] in regs[inds[0]]: top1 += 1

            # Wandb logs
            self.log("train_loss", loss)
            self.log("train_top3", top3 / len(lbl_ims))
            self.log("train_top1", top1 / len(lbl_ims))

            return OD([('loss', loss)])

        def validation_step(self, batch, batch_idx):
            
            """
            
            This function gets batch and batch index and does one validation step and returns validation loss.
            
            Parameters:
            
                batch     - one batch with images, torch dataloader object;
                batch_idx - index of the batch, int.
                
            """

            # Initialize lists to track the metrics 
            cos_sims, cos_unsims, cos_sims_pair, cos_unsims_pair = [], [], [], []
            
            # Get images and labels
            ims, regs = batch
            
            # Get feature maps and pred labels
            lbl_ims = self.model(ims)
            
            # Compute loss
            loss = self.ce_loss(lbl_ims, regs)
            
            # Set initial top3, top1
            top3, top1 = 0, 0
            
            # Go through every predicted label
            for idx, lbl_im in enumerate(lbl_ims):
                
                # Get top3 values and indices
                vals, inds = torch.topk(lbl_im, k=3)
                
                # Top3
                if regs[idx] == regs[inds[0]] or regs[idx] == regs[inds[1]] or regs[idx] == regs[inds[2]]: top3 += 1
                # Top1
                if regs[idx] in regs[inds[0]]: top1 += 1

            # Wandb logs
            self.log("val_loss", loss)
            self.log("val_top3", top3 / len(lbl_ims))
            self.log("val_top1", top1 / len(lbl_ims))

            return OD([('loss', loss), ('val_top3', top3), ('cos_sims', torch.mean(torch.FloatTensor(cos_sims)))])

    def create_model(model_name, num_classes = num_classes):
        
        """ 
        
        This function gets a model name and creates a timm model.
        
        Parameters:
        
            model_name    - a name of a model from timm library, str;
            num_classes   - number of classes in the dataset, int.      
        
        """

        # Get a model
        if model_name in model_dict:
            base_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
            print(f"Model {model_name} with the best weights is successfully loaded!")        
        else:
            assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'
            
        return model

    def train_model(model_name, save_name = None, **kwargs):
        
        """
        
        This function trains the model and returns trained model with its results.
        
        Parameters:
        
            model_name           - a name of the model you want to run, str;
            save_name            - the name that is used for creating the checkpoint and logging directory, str.
        
        """
        
        # Set save name
        if save_name is None: save_name = model_name

        # Initialize trainer
        trainer = pl.Trainer(
            # path to save a trained model
            default_root_dir = os.path.join(sp, save_name),  
            # amp options
            precision = 16, amp_backend = 'native',
            # train epochs
            max_epochs = 300,
            # log steps
            logger = wandb_logger,
            log_every_n_steps = 15,            
            # parallel computing
            strategy = "ddp", accelerator = "gpu", devices = 3, 
            # callbacks
            callbacks = [                
                # Checkpoint
                ModelCheckpoint(
                    # name to save checkpoints
                    filename = '{epoch}-{val_loss:.2f}-{train_loss:.2f}-{val_top1:.2f}', 
                    # save options
                    every_n_train_steps = None, save_top_k = 1,
                    # monitor options
                    save_weights_only = True, mode = "max", monitor = "val_top1" 
                ),
                # Early Stopping
                EarlyStopping(monitor = "val_top1", mode = "max", patience = 20, verbose = True),
                # Log lr every epoch
                LearningRateMonitor("epoch"), 
            ]
        )
        
        # Loggers
        trainer.logger._log_graph = True 
        trainer.logger._default_hp_metric = None

        # Check the availability of the pretrained file
        pretrained_filename = os.path.join(sp, 'rexnet_150_Adam_0.0003', 'Image Retrieval', "1tgu7vtc", "checkpoints")
        # pretrained_filename = pretrained_filename + '/epoch=3-val_loss=4.73-cos_sims=0.91-val_top1=0.48.ckpt'
        
        # if exists
        if os.path.isfile(pretrained_filename):
            print(f"Found pretrained model at {pretrained_filename}, loading...")
            
            # Load from the checkpoint
            model = Model.load_from_checkpoint(pretrained_filename)
        
        # If doesn't exist
        else:
            pl.seed_everything(42) 
            # Initialize model
            model = Model(model_name = model_name, **kwargs)
            # Fit the model with train and validation dataloaders
            trainer.fit(model, train_loader, val_loader)

        return model

    # Train model
    trained_model = train_model(model_name = model_name, optimizer_name = optimizer_name, save_name = f"{model_name}_{optimizer_name}_{lr}", optimizer_hparams = optimizer_hparams)

if __name__ == "__main__":
    
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description = 'Triplet Loss PyTorch Lightning Arguments')
    
    # Add arguments to the parser
    parser.add_argument('-ed', '--expdir', default=None, help='Experiment directory')
    parser.add_argument("-sp", "--save_path", type=str, default='saved_models', help="Path to save trained models")
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
    
    # Parse the arguments from the parser
    args = parser.parse_args() 
    
    # Run the script
    run(args) 
