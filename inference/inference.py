from collections import OrderedDict
import timm, torch, argparse, yaml, os
from softdataset import TRIPLETDATA
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import DataLoader
from collections import OrderedDict as OD
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F
import AutoAugment
from torch.nn import *
from torch import nn
from utils.contrastive_loss import ContrastiveLoss
from utils.square_pad import SquarePad

def run(args):
    
    # Get inference arguments
    path = args.im_path
    inp_size = args.input_size
    bs = args.batch_size
    cache = args.cache
    model_name = args.model_name
    device = args.device
    checkpoint_path = args.checkpoint_path
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(f"\nTraining Arguments:\n{argstr}\n")
    
    # Get dataloader for the inference
    def get_dl(path = path, inp_size = inp_size, bs = bs, cache = cache):        
        
        ''' 
        
        Get path to the images and returns dataloader with transformations applied
        Arguments:
        
        path - path to the dir with images;
        inp_size - input size of images;
        bs - batch_size;
        cache - if True loads pickle file else loads images from the given path.      
        
        '''
        
        # Initialize transformations        
        transformations = {}   
        
        transformations['qry'] = transforms.Compose([
            SquarePad(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])  
        transformations['pos'] = transforms.Compose([
            SquarePad(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])  
        transformations['neg'] = transforms.Compose([
            SquarePad(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        
        # Get the dataset        
        dataset = TRIPLETDATA(path, input_size = inp_size, transform = transformations, cache = cache)  
        datasets = dataset.train_val_test_dataset(dataset)

        # Get test dataset
        test_ds = datasets['test']
        test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=8)
        
        # test_ds = TRIPLETDATA(path, input_size=inp_size, transform = transformations, cache=cache, test=True)  
        # test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=8)

        return test_dl
        
        # test_ds = TRIPLETDATA(path, input_size=inp_size, transform = transformations, cache=cache, test=True)  
        # print(len(test_ds))
        # # datasets = dataset.train_val_test_dataset(dataset)
        # # test_ds = datasets['test']
        # test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=8)

        return test_dl

    def load_checkpoint(checkpoint_path, model_name, pretrained=False, num_classes=0, from_pytorch_lightning=True, conv_input=True):

        ''' 
        
        Loads checkpoint_path from the given path to the directory with the trained model.
        Arguments:
        
        checkpoint_path - path to the dir with the trained model;
        model_name - name of the trained model (name is the same as in the timm library);
        pretrained - creates a model with pretrained weights on ImageNet;
        conv_input - initial convolution layer, default is True;
        from_pytorch_lightning - used to load the trained model from pytorch_lightning.
        If True the model is trained using pytorch_lightning, else with a regular torch library;
        Default is False
        
        '''        
        
        if from_pytorch_lightning: # for a pytorch_lightning model
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            # print(checkpoint['state_dict'].keys())
            if conv_input:                
                base_model = timm.create_model(model_name)
                conv_layer = torch.nn.Sequential(torch.nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1),padding=(1,1), bias=False), 
                 # nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(2, 2),padding=(1,1), bias=False), 
                 # nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                 torch.nn.SiLU(inplace=True))
                model = torch.nn.Sequential(conv_layer, base_model) 
                print(f"Model with conv_input {conv_input}")
            else:
                model = timm.create_model(model_name, num_classes=num_classes)
                print(f"Model with conv_input {conv_input}")

            # create new OrderedDict that does not contain `model.` (for the checkpoint from the pytorch_lightning)
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k.replace("model.", "") # remove `model.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Model {model_name} trained using pytorch lightning checkpoint is successfully loaded!")

        else: # for a regular torch model  
            if pretrained:
                model = timm.create_model(model_name, num_classes=num_classes)
                print(f"Model {model_name} with pretrained weights is successfully loaded!")
            else:
                model = timm.create_model("rexnet_150")
                state_dict = torch.load(checkpoint_path)
                model.load_state_dict(state_dict['state_dict'])
                num_features = model.head.fc.in_features
                model.classifier = Linear(num_features, num_classes) if num_classes > 0 else Identity() 
                print(f"Model {model_name} with the best weights is successfully loaded!")            

        return model

    def inference(model, dataloader, device):

        ''' 
        
        Loads checkpoint from the given path to the directory with the trained model.
        Arguments:
        
        checkpoint - path to the dir with the trained model;
        model_name - name of the trained model (name is the same as in the timm library);
        from_pytorch_lightning - used to load the trained model from pytorch_lightning.
        If True the model is trained using pytorch_lightning, else with a regular torch library;
        Default is False
        
        '''   
        
        # loss_module = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)).to('cuda')
        loss_module = ContrastiveLoss(margin=1.0).to('cuda')
        scores, fms_ims_all, fms_poss_all, losses, classes_all, fms_negs_all = [], [], [], [], [], []
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        model.to(device)
        model.eval()
        print("Obtaining embeddings...")
        for i, batch_all in tqdm(enumerate(dataloader)):
            (ims_all, poss_all, negs_all), clss_all = batch_all
            classes_all.extend(clss_all)
            # if i == 10: # CHANGE HERE
            #     break        
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    fm_ims_batch = model(ims_all.to(device))
                    fm_poss_batch = model(poss_all.to(device))
                    fm_negs_batch = model(negs_all.to(device))
                    loss = loss_module(fm_ims_batch, fm_poss_batch, 1.)
                    losses.append(loss.item())
                    # cos_sims = cos(fm_ims_batch, fm_poss_batch)                
                    fms_ims_all.extend([fm_ims_batch])  
                    fms_poss_all.extend([fm_poss_batch])
                    fms_negs_all.extend([fm_negs_batch])
        print("Embeddings are obtained!")    

        print("Calculating metrics...")
        top1 = 0 
        top3 = 0
        fms_ims_all = torch.stack(fms_ims_all)
        fms_poss_all = torch.stack(fms_poss_all)
        fms_negs_all = torch.stack(fms_negs_all)

        for idx, fm in enumerate(fms_ims_all):
            # print(fm.shape)
            # print(fms_poss_all[idx].shape)
            score = cos(fm, fms_poss_all[idx]) #(bs, fm)
            print(fm.shape)
            print(fms_poss_all[idx].shape)
            score1 = cos(fm, fms_negs_all[idx]) #(bs, fm)
            print((score == score1).sum())
            # print(torch.mean(score).item())
            scores.append(torch.mean(score).item())
            # print(scores)
            vals, inds = torch.topk(cos(fm, fms_poss_all), k=3)
            top3 += len(inds[idx == inds])
            vals, inds = torch.topk(cos(fm, fms_poss_all), k=1)
            top1 += len(inds[idx == inds])
            
        return OD([('loss', np.mean(losses)), ('top1', top1/len(fms_ims_all)), ('top3', top3/len(fms_ims_all)), 
                   ('scores', torch.mean(torch.FloatTensor(scores))), ('normalized_embeddings', fms_ims_all)])
    
    test_dl = get_dl(cache = cache)
    print("Dataloader is ready!")
    model = load_checkpoint(checkpoint_path, model_name, pretrained=False, from_pytorch_lightning=True)
    results = inference(model, test_dl, device)
    print(f"\nTest loss: {results['loss']:.3f}")
    print(f"Test top1: {results['top1']:.3f}")
    print(f"Test top3: {results['top3']:.3f}")
    print(f"Test cos sim scores: {results['scores']:.3f}")
    cc = os.path.splitext(checkpoint_path)[0].split("/")[-1]
    print(f"Checkpoint: {cc}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Inference Arguments')
    parser.add_argument('-ip', '--im_path', type=str, default="/mnt/data/dataset/images-worker_ratio_224", help='Images directory')
    # parser.add_argument('-cp', '--checkpoint_path', type=str, default="/home/ubuntu/workspace/bekhzod/triplet-loss-pytorch/pytorch_lightning/saved_models/model_best.pth.tar", help='Path to the trained model')
    parser.add_argument('-cp', '--checkpoint_path', type=str, default="/home/ubuntu/workspace/bekhzod/triplet-loss-pytorch/pytorch_lightning/saved_models/rexnet_150_Adam_0.0003/lightning_logs/version_12/checkpoints/epoch=26-val_loss=0.03-cos_sims=1.00.ckpt", help='Path to the trained model')
    parser.add_argument("-mn", "--model_name", type=str, default='rexnet_150', help="Model name (from timm library (ex. darknet53))")
    parser.add_argument("-is", "--input_size", type=int, default=(224,224), help="Size of the images")
    parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("-d", "--device", type=str, default='cuda:0', help="GPU device number")
    parser.add_argument("-c", "--cache", type=bool, default=True, help="If False get data from the directory else get data from the saved pickle file")    
    
    args = parser.parse_args() 
    
    run(args)
