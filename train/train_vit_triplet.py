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
from tqdm import tqdm
from torchvision.datasets import ImageFolder
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
    
    # Initialize train dictionaries
    # Model
    model_dict = {}
    model_dict[model_name] = 0 
    
    # Optimizer
    optimizer_hparams={"lr": lr, "weight_decay": wd}
    
    # Initialize transformations
    t = transforms.Compose([transforms.ToTensor()])

    # Get dataset
    ds = TripleDataset('/home/ubuntu/workspace/bekhzod/pytorch-image-models/dataset/real/', '/home/ubuntu/workspace/bekhzod/pytorch-image-models/dataset/sketch/', transform=t)
    print(len(ds))
    
    # Split the dataset into train and validation sets
    tr_ds, val_ds = torch.utils.data.random_split(ds, [5000, 474])
    
    # Initialize train and validation dataloaders
    train_loader = DataLoader(tr_ds, batch_size = bs, shuffle = False, drop_last = False, num_workers = 8)
    val_loader = DataLoader(val_ds, batch_size = bs, shuffle = False, drop_last = False, num_workers = 8)
    
    # Get number of classes
    num_classes = len(ds.classes)
    
    # Function to get feature maps
    def get_fm(fm):
        
        """
        
        This function gets feature map with size (bs, fm_shape, 7, 7)
        applies average pooling and returns feature map with shape (bs, fm_shape).
        
        Argument:
        
            fm - feature map, tensor.
        
        Output:
        
            fm - reshaped feature map, tensor.
        
        """
        
       pool = AvgPool2d((fm.shape[2],fm.shape[3]))
        
       return torch.reshape(pool(fm), (-1, fm.shape[1]))
    
    # Wandb login
    os.system('wandb login 3204eaa1400fed115e40f43c7c6a5d62a0867ed1')     
    
    # Initialize a new run (a new project)
    wandb_logger = WandbLogger(name=f'{model_name}_{datetime.now().strftime("%m/%d/%H:%M:%S")}_triplet_training', project='Triplet-Training')
    print(f"Number of train set images: {len(tr_ds)}")
    print(f"Number of validation set images: {len(val_ds)}")
    print(f"\nTrain dataset has {num_classes} classes")

    # Initialize function to compute cosine similarity
    cos = CosineSimilarity(dim =1 , eps = 1e-6)
    
    # Set labels for the loss function
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
            
            This class gets a model name, optimizer name and hparams and returns trained model (pytorch lightning) with results (dict).
            
            Arguments:
            
                model_name        - name of the model/CNN to run. Used for creating the model (see function below)
                optimizer_name    - name of the optimizer to use. Currently supported: Adam, SGD
                optimizer_hparams - hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
                
            """
            
            super().__init__()
            
            # Save hyperparameters
            self.save_hyperparameters()

            # Create model
            self.model = create_model(model_name)
            
            # Create loss modules
            
            # For triplet 
            self.cos_loss = CosineEmbeddingLoss(margin=0.2) 
            # For classification
            self.ce_loss = CrossEntropyLoss()
            
            # Example tensor
            self.example_input_array = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

        # Forward function of the model    
        def forward(self, inp): return self.model(inp)
        
        # Initialize an optimizer
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
            
            # Other optimizer name
            else:
                assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'
            
            scheduler = MultiStepLR(optimizer = optimizer, milestones = [10, 20, 30, 40, 50], gamma = 0.1, verbose = True)
        
            return [optimizer], [scheduler]
        
        # One training step
        def training_step(self, batch, batch_idx): 
            
            """
            
            This function gets batch and batch index and conducts one step of training process.
            
            Arguments:
            
                batch       - batch of the train dataloader, tensor;
                batch_idx   - index of the batch in the train dataloader, int.
                
            Output:
            
                loss        - loss of training model for the current batch, tensor.
            
            """
            
            # Initialize a list to track cosine similarities
            cos_sims = []
            
            # Get images and labels
            ims, poss, negs, regs = batch['P'], batch['S'], batch['N'], batch['L'] 
            
            # Get predicted feature maps
            fm_ims = self.model(ims)
            fm_poss = self.model(poss)
            fm_negs = self.model(negs)
            
            # Compute loss
            if only_features == True and only_labels == None:
                
                # Cosine embedding loss
                loss_cos_poss = self.cos_loss(fm_ims, fm_poss, labels["pos"].to("cuda")) 
                loss_cos_negs = self.cos_loss(fm_ims, fm_negs, labels["neg"].to("cuda"))
                loss_cos = loss_cos_poss + loss_cos_negs
                loss = loss_cos                 
            
            elif only_features == None and only_labels == True:
                
                # Cross entropy loss
                loss_ce_ims = self.ce_loss(fm_ims, regs)
                loss_ce = loss_ce_ims
                loss = loss_ce
                
            # Set initial top3, top1
            top3, top1 = 0, 0            
            
            # Go through every feature map
            for idx, fm in (enumerate(fm_ims)):
                
                # Compute cosine similarity of the fm of the query image with it's corresponding positive image feature map
                sim_pair = cos(fm_ims[idx].unsqueeze(0), fm_poss[idx].unsqueeze(0)) 
                
                # Compute cosine similarity of the fm of the query image with it's corresponding negative image feature map
                unsim_pair = cos(fm_ims[idx].unsqueeze(0), fm_negs[idx].unsqueeze(0)) 
                
                # Compute cosine similarity of the fm of the query image with every feature map in the mini-batch
                sim = cos(fm_ims[idx].unsqueeze(0), fm_poss) 
                
                # Add to the list
                cos_sims.append(sim)
                
                # Get top3 values and indices
                vals, inds = torch.topk(sim, k=3)
                
                # Compute top1
                if regs[idx] == regs[inds[0]] or regs[idx] == regs[inds[1]] or regs[idx] == regs[inds[2]]: top3 += 1
                
                # Compute top3
                if regs[idx] in regs[inds[0]]: top1 += 1

            # Wand logs
            self.log("train_loss", loss)
            self.log("train_top3", top3 / len(fm_ims))
            self.log("train_top1", top1 / len(fm_ims))

            return OD([('loss', loss)])

        # Validation step of the training process
        def validation_step(self, batch, batch_idx): 
            
            """
            
            This function gets batch and batch index and conducts one step of validation process.
            
            Arguments:
            
                batch           - batch of the train dataloader, tensor;
                batch_idx       - index of the batch in the train dataloader, int.
                
            Outputs:
            
                val_loss        - loss of validation process for the current batch, tensor.
                val_top3        - top3 validation accuracy of the training model for the current batch, tensor.
                cos_sims        - cosine similarities of the current batch, list.
            
            """

            # Initialize lists to track validation process results
            cos_sims, cos_unsims, cos_sims_pair, cos_unsims_pair = [], [], [], []
            
            # Get images and labels
            ims, poss, negs, regs = batch['P'], batch['S'], batch['N'], batch['L'] 
            
            # Get predicted feature maps
            fm_ims = self.model(ims)
            fm_poss = self.model(poss)
            fm_negs = self.model(negs)
            
            # Compute losses
            loss_cos_poss = self.cos_loss(fm_ims, fm_poss, labels["pos"].to("cuda")) 
            loss_cos_negs = self.cos_loss(fm_ims, fm_negs, labels["neg"].to("cuda"))
            loss_cos = loss_cos_poss + loss_cos_negs
            loss = loss_cos
            
            # Set initial top3, top1 values            
            top3, top1 = 0, 0
            
            # Go through every feature map
            for idx, fm in enumerate(fm_ims):
                
                # Compute cosine similarity of the fm of the query image with it's corresponding positive image feature map
                sim_pair = cos(fm_ims[idx].unsqueeze(0), fm_poss[idx].unsqueeze(0)) 
                
                # Compute cosine similarity of the fm of the query image with it's corresponding negative image feature map
                unsim_pair = cos(fm_ims[idx].unsqueeze(0), fm_negs[idx].unsqueeze(0)) 
                
                # Compute cosine similarity of the fm of the query image with every feature map in the mini-batch
                sim = cos(fm_ims[idx].unsqueeze(0), fm_poss) 
                
                # Add to the cos_similarities list
                cos_sims_pair.append(sim_pair)
                
                # Add to the cos_unsimilarities list
                cos_unsims_pair.append(unsim_pair)
                
                # Get top3 values and indices
                vals, inds = torch.topk(sim, k=3)

                # Compute top3
                if regs[idx] == regs[inds[0]] or regs[idx] == regs[inds[1]] or regs[idx] == regs[inds[2]]: top3 += 1
                    
                # Compute top1
                if regs[idx] in regs[inds[0]]: top1 += 1
                
            # Wandb logs
            self.log("val_loss", loss)
            self.log("cos_sims", torch.mean(torch.FloatTensor(cos_sims_pair)).item())
            self.log("cos_unsims", torch.mean(torch.FloatTensor(cos_unsims_pair)).item())
            self.log("val_top3", top3 / len(fm_ims))
            self.log("val_top1", top1 / len(fm_ims))

            return OD([('loss', loss), ('val_top3', top3), ('cos_sims', torch.mean(torch.FloatTensor(cos_sims)))])

    def create_model(model_name, num_classes = num_classes):
        
        """ 
        
        This function gets model name and creates a timm model.
        
        Arguments:
        
            model_name  - name of the model in timm library, str;
            conv_input  - option for the the input to pass convolution layer first, bool;
            num_classes - number of classes in the dataset.
            
        Output:
        
            model       - a created model from timm library.
        
        """

        # Search for the model name in the dict
        if model_name in model_dict:
            
            # Get the model 
            base_model = timm.create_model(model_name, pretrained = True, num_classes = num_classes)
            
            # Change model head to Identity to get feature maps
            base_model.head = Identity()
            print(f"Model {model_name} with the best weights is successfully loaded!")        
        else:
            assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'
            
        return model

    def train_model(model_name, save_name = None, **kwargs):
        
        """
        
        This function trains the model and returns trained model with its results.
        
        Arguments:
        
            model_name - name of the model in timm library, str;
            save_name  - name for the saved model, str.
            
        Outputs:
        
            model      - a trained model, timm model;
            result     - train evaluation results, dictionary.
            
        """
        
        # Initialize save name
        if save_name is None:
            save_name = model_name

        # Pytorch lightning trainer
        trainer = pl.Trainer(
            
            # root to save chechpoints
            default_root_dir = os.path.join(sp, save_name),
            # amp options
            precision = 16, amp_backend = 'native',
            
            # epochs to train
            max_epochs = 300,
            
            # logger options
            logger = wandb_logger, log_every_n_steps = 15,

            # parallel computing options
            strategy = "ddp", accelerator = "gpu", devices = 3, 
            
            # Train callbacks
            callbacks=[
                # Save checkpoint
                ModelCheckpoint(
                    # Name for the checkpoint
                    filename='{epoch}-{val_loss:.2f}-{cos_sims:.2f}-{val_top1:.2f}', 
                    # Saving options
                    every_n_train_steps = None, save_top_k = 1, save_weights_only = True,
                    # Tracking metrics
                    mode = "max", monitor = "cos_sims" 
                ), 
                
                # Early stopping
                EarlyStopping(
                    # Tracking metrics
                    monitor = "cos_sims", mode = "max", patience = 10, verbose = True),
                
                
                LearningRateMonitor("epoch"), # Log learning rate every epoch
            ]
        )
        
        # Log options
        trainer.logger._log_graph = True  
        trainer.logger._default_hp_metric = None 

        # Check pretrained file
        pretrained_filename = os.path.join(sp, 'rexnet_150_Adam_0.0003', 'Image Retrieval', "1tgu7vtc", "checkpoints")
        # pretrained_filename = pretrained_filename + '/epoch=3-val_loss=4.73-cos_sims=0.91-val_top1=0.48.ckpt'
        
        # If exists
        if os.path.isfile(pretrained_filename):
            print(f"Found pretrained model at {pretrained_filename}, loading...")
            
            # Load weights from the pretrained file
            model = Model.load_from_checkpoint(pretrained_filename)
        
        # Other case
        else:
            # Set seed
            pl.seed_everything(42) 
            
            # Initialize model
            model = Model(model_name = model_name, **kwargs)
            
            # Fit train and validation dataloaders to the model
            trainer.fit(model, train_loader, val_loader)

        return model
    
    # Start training 
    trained_model = train_model(model_name = model_name, optimizer_name = optimizer_name,
    save_name = f"{model_name}_{optimizer_name}_{lr}", optimizer_hparams = optimizer_hparams)

if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description='Triplet Loss PyTorch Lightning Arguments')
    
    # Add arguments to the parser
    parser.add_argument('-ed', '--expdir', default = None, help = 'Experiment directory')
    parser.add_argument("-sp", "--save_path", type = str, default = 'saved_models', help = "Path to save trained models")
    parser.add_argument("-bs", "--batch_size", type = int, default = 32, help = "Batch size")
    parser.add_argument("-d", "--device", type = str, default = 'cuda:1', help = "GPU device number")
    parser.add_argument("-ip", "--ims_path", type = str, default = '/home/ubuntu/workspace/dataset/test_dataset_svg/pass_images_dataset_spec49', help = "Path to the images")
    parser.add_argument("-mn", "--model_name", type = str, default = 'swin_s3_base_224', help = "Model name (from timm library (ex. darknet53, ig_resnext101_32x32d))")
    parser.add_argument("-on", "--optimizer_name", type = str, default = 'Adam', help = "Optimizer name (Adam or SGD)")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-5, help = "Learning rate value")
    parser.add_argument("-wd", "--weight_decay", type = float, default = 1e-6, help = "Weight decay value")
    parser.add_argument("-ofm", "--only_feature_embeddings", type = bool, default = True,
                        help="If True trains the model using only triplet loss and and return feature embeddings (if both otl and ofm are True uses two loss functions simultaneously)")
    parser.add_argument("-otl", "--only_target_labels", type = bool, default = None,
                        help="If True trains the model using only cross entropy and and return predicted labels (if both otl and ofm are True uses two loss functions simultaneously)")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args)
