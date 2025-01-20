import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils_SSL import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval,Dataset_ASVspoof2019_eval
from model_ppl import Model
from core_scripts.startup_config import set_random_seed
import matplotlib.pyplot as plt
import eval_metrics as em

# Modified from https://github.com/TakHemlata/SSL_Anti-spoofing/blob/main/main_SSL_LA.py


class ppl_loss(nn.Module):
    def __init__(self, feat_dim=128, r_real=0.5, r_fake=0.2, alpha_g=20.0, alpha_s=20.0):
        super(ppl_loss, self).__init__()
        # self.feat_dim = feat_dim
        self.alpha_g = alpha_g
        self.alpha_s = alpha_s
        self.r_real = r_real
        self.r_fake = r_fake
        self.fc = nn.Linear(160,feat_dim,bias=False)
        self.center = nn.Parameter(torch.randn(1, feat_dim), requires_grad=False)
        nn.init.xavier_uniform_(self.center)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """  
        x = self.fc(x)
        w = nn.functional.normalize(self.center, p=2, dim=1)
        x = nn.functional.normalize(x, p=2, dim=1)
        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()
        # # for monitoring the training process
        # theta = torch.acos(torch.clamp(scores, -1.0 + 1e-7, 1.0 - 1e-7))
        # theta_med_tar = torch.mean(theta[labels == 1].squeeze(1))
        # theta_med_non = torch.mean(theta[labels == 0].squeeze(1))
        
        
        if labels==None:
            # evaluation mode
            return output_scores.squeeze(1),None
        else:
            scores[labels == 1] = self.alpha_g*(self.r_real - scores[labels == 1]) 
            scores[labels == 0] = self.alpha_s*(scores[labels == 0] - self.r_fake)

            loss = (self.softplus(scores)).nanmean() # OC-Softmax
            
            
            if torch.sum(scores[labels == 0]) != 0:
                # ## Euclidean distance based push ##
                # spo_center = x[labels == 0].mean(0).unsqueeze(0)
                # spo_center = nn.functional.normalize(spo_center, p=2, dim=1)
                # euclidean_distance = nn.functional.pairwise_distance(x[labels == 0], spo_center) #/ torch.sqrt(torch.tensor(w.shape[1]))
                # attack_loss = torch.pow(torch.clamp(  0.5-euclidean_distance, min=0.0), 2) 
                
                # ## Angle based push ##
                spo_center = x[labels == 0].mean(0).unsqueeze(0)
                spo_center = nn.functional.normalize(spo_center, p=2, dim=1)
                dist_score = x[labels == 0] @ spo_center.transpose(0,1)
                dist_angle = torch.acos(torch.clamp(dist_score, -1.0 + 1e-7, 1.0 - 1e-7)).squeeze()
                T1 = torch.clamp(  0.5-dist_angle, min=0.0)
                attack_loss = torch.pow(T1, 2) 

                loss_total = loss +  attack_loss.nanmean() # PPL

            return output_scores.squeeze(1), loss_total
    
    
    
def evaluate_accuracy(dev_loader, model, device,ppl_layer):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    ppl_layer.eval()

    idx_loader, score_loader = [], []

    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
    
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)
            scores, pploss = ppl_layer(batch_out, batch_y) 
            val_loss += (pploss.item() * batch_size)
            idx_loader.append(batch_y)
            score_loader.append(scores.squeeze())

    scores = torch.cat(score_loader, 0).data.cpu().numpy()
    labels = torch.cat(idx_loader, 0).data.cpu().numpy()

    val_eer = em.compute_eer(scores[labels == 1], scores[labels == 0])[0]
    val_loss /= num_total
    return val_loss, val_eer


def produce_evaluation_file(dataset, model, device, save_path,ppl_layer=None):
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False) # batch_size=10 orj.
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    ppl_layer.eval()
    fname_list = []
    key_list = []
    score_list = []
    with torch.no_grad():
        for batch_x,utt_id in data_loader:
            fname_list = []
            score_list = []
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
    
            batch_out = model(batch_x)
            batch_score, pploss = ppl_layer(batch_out, None) 

            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
    
            with open(save_path, 'a+') as fh:
                for f, cm in zip(fname_list,score_list):
                    fh.write('{} {}\n'.format(f, cm))
            fh.close()
    print('Scores saved to {}'.format(save_path))

def train_epoch(train_loader, model, lr,optim, device,ppl_layer, optim_ppl=None):
    running_loss = 0

    num_total = 0.0

    model.train()
    ppl_layer.train()

    for batch_x, batch_y in train_loader:

        batch_size = batch_x.size(0)
        num_total += batch_size
    
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
            
        scores, pploss = ppl_layer(batch_out, batch_y) 

        optim_ppl.zero_grad()
        optim.zero_grad()
        
        pploss.backward()
        
        optim_ppl.step()
        optim.step()
        running_loss += (pploss.item() * batch_size) #ppl
    running_loss /= num_total    

    return running_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/your/path/to/data/ASVspoof_database/LA/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- LA
    %      |- ASVspoof2021_LA_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac

 

    '''

    parser.add_argument('--protocols_path', type=str, default='database/', help='Change with path to user\'s LA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt
    %      |- ASVspoof2019.LA.cm.train.trn.txt

    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=24)#default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default= 1e-6) #default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='ppl') 
    # model
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')

    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
                        
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='LA',choices=['LA', 'PA','DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--eval19', action='store_true', default=False,
                        help='eval19 mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True,
                        help='use cudnn-deterministic? (default true)')

    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False,
                        help='use cudnn-benchmark? (default false)')


    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=5,
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters
    parser.add_argument('--nBands', type=int, default=5,
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20,
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000,
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100,
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000,
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10,
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100,
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0,
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0,
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5,
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20,
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5,
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10,
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2,
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10,
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40,
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')

    ##===================================================Rawboost data augmentation ======================================================================#


    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()

    #make experiment reproducible
    set_random_seed(args.seed, args)

    track = args.track

    assert track in ['LA', 'PA','DF'], 'Invalid track given'

    #database
    prefix      = 'ASVspoof_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)

    #define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    model = Model(args,device)

    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =model.to(device)
    print('nb_params:',nb_params)

    
    ## ppl optimizer
    ppl_layer = ppl_loss(feat_dim=2,  r_real=0.5, r_fake=0.2, alpha_g=20,alpha_s=20)
    ppl_layer =ppl_layer.to(device)

    
 #%%   
    ### load trained model & PPL for evaluation
    # ppl_layer.load_state_dict(torch.load('path_to_ppl_layer',map_location=device, weights_only=True))
    # print('ppl loaded')
    # ppl_layer =ppl_layer.to(device)
    
    # ##Load trained model & PPL
    # if args.model_path:
    #     model.load_state_dict(torch.load(args.model_path,map_location=device))
    #     print('Model loaded : {}'.format(args.model_path))
    # model =model.to(device)  

#%%
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    ppl_lr = 1e-6
    optim_ppl = torch.optim.Adam([{'params':ppl_layer.parameters(), 'lr':ppl_lr, 'weight_decay':args.weight_decay}])
   
    #evaluation
    if args.eval:
        file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.eval.trl.txt'.format(prefix,prefix_2021)),is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(args.track)))
        produce_evaluation_file(eval_set, model, device, args.eval_output,ppl_layer=ppl_layer)
        sys.exit(0)
        
    #evaluation of 2019 LA data
    if args.eval19:
        key_eval,file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.eval.trl.txt'.format(prefix,prefix_2019)),is_train=True,is_eval=False)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_ASVspoof2019_eval(args,list_IDs = file_eval,base_dir = os.path.join(args.database_path+'{}_{}_eval/'.format(prefix_2019.split('.')[0],args.track)), labels=key_eval,algo=args.algo)
        produce_evaluation_file(eval_set, model, device, args.eval_output,ppl_layer=ppl_layer)
        sys.exit(0)




    # define train dataloader
    d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.train.trn.txt'.format(prefix,prefix_2019)),is_train=True,is_eval=False)

    print('no. of training trials',len(file_train))

    train_set=Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'{}_{}_train/'.format(prefix_2019.split('.')[0],args.track)),algo=args.algo)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True, pin_memory=True)
    del train_set,d_label_trn

    # define validation dataloader
    d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.dev.trl.txt'.format(prefix,prefix_2019)),is_train=False,is_eval=False)

    print('no. of validation trials',len(file_dev))

    dev_set = Dataset_ASVspoof2019_train(args,list_IDs = file_dev,
		labels = d_label_dev,
		base_dir = os.path.join(args.database_path+'{}_{}_dev/'.format(prefix_2019.split('.')[0],args.track)),algo=args.algo)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False,pin_memory=True)
    del dev_set,d_label_dev


    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
      
        running_loss = train_epoch(train_loader,model, args.lr,optimizer, device, ppl_layer=ppl_layer, optim_ppl=optim_ppl)
        val_loss, val_eer = evaluate_accuracy(dev_loader, model, device, ppl_layer)

        with open(os.path.join( model_save_path , "train.log"), "a") as log:
            log.write("Epoch "+str(epoch) + "\t"  +"train_loss= "+ str(running_loss)+ "\t" +"val_loss= "+ str(val_loss)+ "\t"+ "val_eer= "+ str(val_eer*100) + "\n")
        print('\n{} - {} - {} - EER={} '.format(epoch,running_loss,val_loss, val_eer*100))
        if ((epoch+1) % 1) == 0:
            torch.save(model.state_dict(), os.path.join(
                model_save_path, 'epoch_{}.pth'.format(epoch)))
            torch.save(ppl_layer.state_dict(), os.path.join(
                model_save_path, 'epoch_ppl{}.pth'.format(epoch)))