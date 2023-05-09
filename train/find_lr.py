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
from sketch_dataset import SketchyImageDataset
from collections import namedtuple as NT
from softdataset import TripletImageDataset, data_split
from original_dataset import OriginalImageDataset, data_split
from tqdm import tqdm
import torchvision.transforms.functional as TF
from contrastive_loss import ContrastiveLoss

def run(args):
    
    # Get the arguments
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
    only_features = args.only_feature_embeddings
    only_labels = args.only_target_labels
    
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(f"\nTraining Arguments:\n{argstr}")
    
    optimizer_hparams={"lr": lr}
    model_dict[model_name] = 0 
    os.system('wandb login 3204eaa1400fed115e40f43c7c6a5d62a0867ed1')
    
    # Set the transformations
    transformations = {}   
    
    # Transformations for query images
    transformations['qry'] = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    # Transformations for positive sketch images
    transformations['pos'] = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])  
    
    # Transformations for negative sketch images
    transformations['neg'] = transforms.Compose([transforms.Resize((224,224)), AutoAugment.ImageNetPolicy(), transforms.ToTensor()])
    
    # Path to the json file
    out_path = "data/sketchy_database_256_soft_split_cat.json"
    
    # Get training, validation, and test datasets
    tr_ds = SketchyImageDataset(data_dir = path, transform_dic = transformations, random = True, trainval_json = out_path, trainval = 'train', load_images = False)
    val_ds = SketchyImageDataset(data_dir = path, transform_dic = transformations, random = True, trainval_json = out_path, trainval = 'val', load_images = False)
    test_ds = SketchyImageDataset(data_dir = path, transform_dic = transformations, random = True, trainval_json = out_path, trainval = 'test', load_images = False)
    
    wandb_logger = WandbLogger(name = f'{model_name}_{datetime.now().strftime("%m/%d/%H:%M:%S")}_triplet_training', project = 'Train-test-LR')
    num_classes = tr_ds.get_cat_length()
    print(f"Number of train set images: {len(tr_ds)}")
    print(f"Number of validation set images: {len(val_ds)}")
    print(f"Number of test set images: {len(test_ds)}")
    print(f"\nTrain dataset has {num_classes} classes")
    print(f"Validation dataset has {val_ds.get_cat_length()} classes")
    print(f"Test dataset has {test_ds.get_cat_length()} classes")
    
    # Initialize cosine similarity
    cos = CosineSimilarity(dim = 1, eps = 1e-6)
    
    # Get train, validation, and test dataloaders
    train_loader = DataLoader(tr_ds, batch_size = bs, shuffle = True, drop_last = False, num_workers = 8)
    val_loader = DataLoader(val_ds, batch_size = bs, shuffle = True, drop_last = True, num_workers = 8)
    test_loader = DataLoader(test_ds, batch_size = bs, shuffle = True, drop_last = False, num_workers = 8)  
    
    # Set labels for the loss functions
    labels = {"con_pos": torch.tensor(1.).unsqueeze(0),
              "con_neg": torch.tensor(0.).unsqueeze(0),
              "cos_pos": torch.tensor(1.).unsqueeze(0),
              "cos_neg": torch.tensor(-1.).unsqueeze(0)}
    
    alpha, eps = 1, 5
    
    # Similarity score booster function
    def cos_sim_score(score, eps, alpha, mode):
        if mode == "for_pos":
            if score < 0.3: return (score + eps) / (eps + eps*alpha)
            else: return (score + eps) / (eps + alpha)
        # elif score > 0.5:
        elif mode == "for_neg": return (score + (alpha / eps)) / (2*eps)
    
    assert only_features or only_labels, "Please choose at least one loss function to train the model (triplet loss or crossentropy loss)"
    if only_features and only_labels:
        print("\nTrain using triplet loss and crossentropy loss\n")
    elif only_features == True and only_labels == None:
        print("\nTrain using only triplet loss\n")                
    elif only_features == None and only_labels == True:
        print("\nTrain using only crossentropy loss\n")      
    
    # Initialize model
    class Model(pl.LightningModule):

        def __init__(self, model_name,  optimizer_name, optimizer_hparams):
            
            """
            
            This class gets a model name, optimizer name and hparams and returns trained model (pytorch lightning) with results (dict).
            
            Parameters:
            
                model_name        - name of the model/CNN to run. Used for creating the model (see function below)
                optimizer_name    - name of the optimizer to use. Currently supported: Adam, SGD
                optimizer_hparams - hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
                
            """
            
            super().__init__()
            # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
            self.save_hyperparameters()

            # Create model
            self.model = create_model(model_name)
            # Create loss modules and change them to cuda
            self.cos_loss = CosineEmbeddingLoss(margin=0.3).to('cuda')
            self.con_loss = ContrastiveLoss(margin=0.3).to('cuda')
            self.ce_loss = CrossEntropyLoss().to('cuda')
            
            # Example input for visualizing the graph in Tensorboard
            self.example_input_array = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
            if self.hparams.optimizer_hparams['lr'] is not None:
                self.hparams["lr"] = self.hparams.optimizer_hparams["lr"]

        def forward(self, inp):
            
            # Function to convert dictionary to namedtuple
            def dict_to_namedtuple(dic):
                return NT('GenericDict', dic.keys())(**dic)
            
            dic = {}                        
            fm = self.model.forward_features(inp)
            pool = AvgPool2d((fm.shape[2],fm.shape[3]))
            lbl = self.model.head(fm)
            dic["feature_map"] = torch.reshape(pool(fm), (-1, fm.shape[1]))
            dic["class_pred"] = lbl
            out = dict_to_namedtuple(dic)
            
            return out
        
        def configure_optimizers(self):
            
            
            """
            
            This function initializes optimizer and scheduler.
            
            Outputs:
            
                optimizer - optimizer to update trainable parameters of the model;
                scheduler - scheduler of the optimizer.
            
            """
            
            assert self.hparams.optimizer_name in ["Adam", "SGD"], "Currently only Adam and SGD optimizer are supported."
            if self.hparams.optimizer_name == "Adam":  optimizer = torch.optim.AdamW(self.parameters(), self.hparams.lr)
            elif self.hparams.optimizer_name == "SGD": optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
            
            milestones = [5,10,15,20,25,30,35,40,45,50]
            gamma = 0.1
            scheduler = MultiStepLR(optimizer = optimizer, milestones = milestones, gamma = gamma, verbose = True)
        
            return [optimizer], [scheduler]
        
        def training_step(self, batch, batch_idx): 
            
            """
            
            This function gets batch and batch index and conducts one step of training process.
            
            Parameters:
            
                batch       - batch of the train dataloader, tensor;
                batch_idx   - index of the batch in the train dataloader, int.
                
            Output:
            
                loss        - loss of training model for the current batch, tensor.
            
            """
            
            cos_sims = []
            ims, poss, negs, clss, regs = batch['qry'], batch['pos'][0], batch['neg'][0], batch['cat_idx'], batch['prod_idx']
            
            # Get feature maps and pred labels of the input images
            out_ims = self(ims) 
            # Get feature maps [0] and predicted labels [1]
            fm_ims, lbl_ims = out_ims[0], out_ims[1] 
            
            # Get feature maps and pred labels of the positive images
            out_poss = self(poss)
            # Get feature maps [0] and predicted labels [1]
            fm_poss, lbl_poss = out_poss[0], out_poss[1] 
            
            # Get feature maps and pred labels of the negative images
            out_negs = self(negs)
            # Get feature maps [0] and predicted labels [1]
            fm_negs, lbl_negs = out_negs[0], out_negs[1] 
            
            # Compute loss
            if only_features and only_labels:
                
                # Cosine embedding loss
                loss_cos_poss = self.cos_loss(fm_ims.to("cuda"), fm_poss.to("cuda"), labels["cos_pos"].to("cuda")) 
                loss_cos_negs = self.cos_loss(fm_ims.to("cuda"), fm_negs.to("cuda"), labels["cos_neg"].to("cuda"))
                loss_cos = loss_cos_poss + loss_cos_negs
                
                # Contrastive loss
                loss_con_poss = self.con_loss(fm_ims.to("cuda"), fm_poss.to("cuda"), labels["con_pos"].to("cuda")) 
                loss_con_negs = self.con_loss(fm_ims.to("cuda"), fm_negs.to("cuda"), labels["con_neg"].to("cuda"))
                loss_con = loss_con_poss + loss_con_negs
                
                # Cross entropy loss
                loss_ce_ims = self.ce_loss(lbl_ims, clss)
                loss_ce_poss = self.ce_loss(lbl_poss, clss)
                loss_ce = loss_ce_ims + loss_ce_poss
                loss = loss_cos + loss_con + loss_ce
                
            # Cosine embedding loss only               
            elif only_features == True and only_labels == None:
                loss_cos_poss = self.cos_loss(fm_ims, fm_poss, labels["pos"].to("cuda")) 
                loss_cos_negs = self.cos_loss(fm_ims, fm_negs, labels["neg"].to("cuda"))
                loss_cos = loss_cos_poss + loss_cos_negs
                loss = loss_cos         
                
            # Crossentropy loss only               
            elif only_features == None and only_labels == True:
                loss_ce_ims = self.ce_loss(lbl_ims, regs)
                loss_ce = loss_ce_ims
                loss = loss_ce
                
            # Compute top3 and top1
            top3, top1 = 0, 0            
            for idx, fm in (enumerate(fm_ims)):
                sim = cos(fm_ims[idx].unsqueeze(0), fm_poss[idx]) 
                cos_sims.append(sim)
                vals, inds = torch.topk(lbl_ims[idx], k = 3)
                if regs[idx] in inds: top3 += 1
                if regs[idx] in inds[0]: top1 += 1

            return OD([('loss', loss)]) 

        def validation_step(self, batch, batch_idx):  
            
            """
            
            This function gets batch and batch index and conducts one step of validation process.
            
            Parameters:
            
                batch           - batch of the train dataloader, tensor;
                batch_idx       - index of the batch in the train dataloader, int.
                
            Outputs:
            
                val_loss        - loss of validation process for the current batch, tensor.
                val_top3        - top3 validation accuracy of the training model for the current batch, tensor.
                cos_sims        - cosine similarities of the current batch, list.
            
            """

            cos_sims = []
            ims, poss, negs, clss, regs = batch['qry'], batch['pos'][0], batch['neg'][0], batch['cat_idx'], batch['prod_idx']

            # Get feature maps and pred labels of the input images
            out_ims = self(ims) 
            # Get feature maps [0] and predicted labels [1]
            fm_ims, lbl_ims = out_ims[0], out_ims[1] 
            
            # Get feature maps and pred labels of the positive images
            out_poss = self(poss)
            # Get feature maps [0] and predicted labels [1]
            fm_poss, lbl_poss = out_poss[0], out_poss[1] 
            
            # Get feature maps and pred labels of the negative images
            out_negs = self(negs)
            # Get feature maps [0] and predicted labels [1]
            fm_negs, lbl_negs = out_negs[0], out_negs[1]
            
            # Compute loss
            if only_features and only_labels:
                
                # Cosine embedding loss
                loss_cos_poss = self.cos_loss(fm_ims.to("cuda"), fm_poss.to("cuda"), labels["cos_pos"].to("cuda")) 
                loss_cos_negs = self.cos_loss(fm_ims.to("cuda"), fm_negs.to("cuda"), labels["cos_neg"].to("cuda"))
                loss_cos = loss_cos_poss + loss_cos_negs
                
                # Contrastive loss
                loss_con_poss = self.con_loss(fm_ims.to("cuda"), fm_poss.to("cuda"), labels["con_pos"].to("cuda")) 
                loss_con_negs = self.con_loss(fm_ims.to("cuda"), fm_negs.to("cuda"), labels["con_neg"].to("cuda"))
                loss_con = loss_con_poss + loss_con_negs
                
                # Cross entropy loss
                loss_ce_ims = self.ce_loss(lbl_ims, clss)
                loss_ce_poss = self.ce_loss(lbl_poss, clss)
                loss_ce = loss_ce_ims + loss_ce_poss
                loss = loss_cos + loss_con + loss_ce
                
            # Cosine embedding loss only
            elif only_features == True and only_labels == None:
                loss_cos_poss = self.cos_loss(fm_ims, fm_poss, labels["pos"].to("cuda")) 
                loss_cos_negs = self.cos_loss(fm_ims, fm_negs, labels["neg"].to("cuda"))
                loss_cos = loss_cos_poss + loss_cos_negs
                loss = loss_cos    
                
            # Cross entropy loss only
            elif only_features == None and only_labels == True:
                loss_ce_ims = self.ce_loss(lbl_ims, regs)
                loss_ce = loss_ce_ims
                loss = loss_ce              
            
            # Compute top3 and top1            
            top3, top1 = 0, 0
            
            for idx, fm in (enumerate(fm_ims)):
                sim = cos(fm_ims[idx].unsqueeze(0), fm_poss[idx]) 
                cos_sims.append(sim)
                vals, inds = torch.topk(lbl_ims[idx], k=3)
                if regs[idx] in inds: top3 += 1
                if regs[idx] in inds[0]: top1 += 1

            # Logs the loss per epoch to tensorboard (weighted average over batches)
            self.log("val_loss", loss)
            self.log("cos_sims", cos_sim_score(torch.mean(torch.FloatTensor(cos_sims)).item(), eps, alpha, mode='for_pos'))
            self.log("val_top3", top3 / len(fm_ims))
            self.log("val_top1", top1 / len(fm_ims))

            return OD([('loss', loss), ('val_top3', top3), ('cos_sims', torch.mean(torch.FloatTensor(cos_sims)))])

    def create_model(model_name, conv_input=False, num_classes=num_classes):
        
        """ 
        
        This function gets model name and creates a timm model.
        
        Parameters:
        
            model_name  - name of the model in timm library, str;
            conv_input  - option for the the input to pass convolution layer first, bool;
            num_classes - number of classes in the dataset, int.
            
        Output:
        
            model       - created model from timm library.
        
        """

        if model_name in model_dict:
            base_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
            print(f"Model {model_name} with the best weights is successfully loaded!")        
            if conv_input:                
                conv_layer = Sequential(Conv2d(3, 3, kernel_size = (3, 3), stride = (1, 1),padding = (1,1), bias = False),  SiLU(inplace = True))
                model = Sequential(conv_layer, base_model)  
            else: model = base_model
        else: assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'
            
        return model

    def train_model(model_name, save_name = None, **kwargs):
        
        """
        
        This function trains the model and returns trained model with its results.
        
        Parameters:
        
            model_name - name of the model in timm library, str;
            save_name  - name for the saved model, str.
            
        Outputs:
        
            model      - a trained model, timm model;
            result     - train evaluation results, dictionary.
            
        """
        
        if save_name is None: save_name = model_name

        # Create a PyTorch Lightning trainer with the generation callback
        trainer = pl.Trainer(
            # path to the folder to save models
            default_root_dir=os.path.join(sp, save_name),  
            # number of epochs to train the model
            max_epochs=300,
            # log steps
            log_every_n_steps=15,
            # logger name
            logger=wandb_logger,
            # auto lr finder option
            auto_lr_find=True,
#             fast_dev_run=True,
            callbacks=[
                # Model Checkpoint
                ModelCheckpoint(
                    # name for the checkpoint
                    filename='{epoch}-{val_loss:.2f}-{cos_sims:.2f}-{val_top1:.2f}', 
                    # save frequency
                    every_n_train_steps = None,
                    # option to save the best model
                    save_top_k=1,
                    # option to save weights only
                    save_weights_only=True, 
                    # metric monitor options
                    mode="max", monitor="val_top1" 
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
            model = Model(model_name=model_name,  optimizer_name=optimizer_name, optimizer_hparams=optimizer_hparams)
            lr_finder = trainer.tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            model.hparams.lr = lr_finder.suggestion()
            trainer.fit(model, train_loader, val_loader)

        # Test best model on validation and test set
        test_result = trainer.test(model, dataloaders=test_loader, verbose=True)

        result = {"test_loss": test_result[0]["test_loss"], 
                  "test_scores": test_result[0]["test_sim_scores"],
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
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description='Triplet Loss PyTorch Lightning Arguments')
    
    # Add Arguments to the Parser
    parser.add_argument('-ed', '--expdir', default=None, help='Experiment directory')
    parser.add_argument("-sp", "--save_path", type=str, default='saved_models', help="Path to save trained models")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-d", "--device", type=str, default='cuda:1', help="GPU device number")
    parser.add_argument("-ip", "--ims_path", type=str, default='/home/ubuntu/workspace/dataset/sketchy_database_256', help="Path to the images")
    parser.add_argument("-is", "--input_size", type=int, default=(224, 224), help="Size of the images")
    parser.add_argument("-mn", "--model_name", type=str, default='rexnet_150', help="Model name (from timm library (ex. darknet53, ig_resnext101_32x32d))")
    parser.add_argument("-on", "--optimizer_name", type=str, default='Adam', help="Optimizer name (Adam or SGD)")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-1, help="Learning rate value")
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-5, help="Weight decay value")
    parser.add_argument("-ofm", "--only_feature_embeddings", type=bool, default=True,
                        help="If True trains the model using only triplet loss and and return feature embeddings (if both otl and ofm are True uses two loss functions simultaneously)")
    parser.add_argument("-otl", "--only_target_labels", type=bool, default=True,
                        help="If True trains the model using only cross entropy and and return predicted labels (if both otl and ofm are True uses two loss functions simultaneously)")
    
    # Parse the Arguments
    args = parser.parse_args() 
    
    # Run the script
    run(args)
