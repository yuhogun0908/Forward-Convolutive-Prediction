import sys
sys.path.append('./')
import argparse
# from model import MYNET, ResidualBlock, Share_BN_ResidualBlock
import yaml

# from trainer import ModelTrainer, ModelTester
from torch.utils.data import DataLoader
import trainer
# from utils.setup import setup_solver
# from utils.loss import create_criterion
import pdb
import os
import pickle
from model import MISO_1, MISO_2, MISO_3
import torch
from trainer import Trainer_Separate, Trainer_Enhance, Trainer_Beamforming
# Blind Source Separation by using NN
# 1. Feature Extractor
# 2. Build dataloader 
# 3. Train
# 4. Test
#
# Reference
# Z.Q, Wang. "Multi-microphone Complex Spectral Mapping for Utterance-wise and Continuous Speech Separation", IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 29, 2021

def run(args,config):
    fs = config[args.dataset]['fs']; chunk_time = config[args.dataset]['chunk_time']; least_time = config[args.dataset]['least_time']
    num_spks = config[args.dataset]['num_spks']
    num_ch = config[args.dataset]['num_ch']
    if args.mode == 'Extraction':
        if args.dataset == 'REVERB_2MIX':
            from dataloader.REVERB_2MIX import main_reverb
            fs = config['REVERB_2MIX']['fs']; chunk_time = config['REVERB_2MIX']['chunk_time']; least_time = config['REVERB_2MIX']['least_time']
            num_spks = config['REVERB_2MIX']['num_spks']
            num_ch = config['REVERB_2MIX']['num_ch']
            scp_list = config['REVERB_2MIX']['scp_list']; 
            # select the type of data to save                    
            if config['REVERB_2MIX']['select_mode'] == 1:  # save train 
                tr_wave_list = config['REVERB_2MIX']['tr_parent_wave_list']; save_tr_pickle_dir = config['REVERB_2MIX']['saved_tr_pickle_dir']
                main_reverb('Train',fs, chunk_time, least_time,num_spks,scp_list, tr_wave_list,save_tr_pickle_dir)  
            elif config['REVERB_2MIX']['select_mode'] == 2: # save development
                dt_wave_list = config['REVERB_2MIX']['dt_parent_wave_list']; save_dt_pickle_dir = config['REVERB_2MIX']['saved_dt_pickle_dir']
                main_reverb('Development',fs, chunk_time, least_time,num_spks,scp_list, dt_wave_list,save_dt_pickle_dir)  
            elif config['REVERB_2MIX']['select_mode'] == 3: # save test 
                pdb.set_trace()
        if args.dataset == 'RIR_mixing':
            from dataloader.RIR_mixing import main_rirmixing
            fs = config['RIR_mixing']['fs']; chunk_time = config['RIR_mixing']['chunk_time']; least_time = config['RIR_mixing']['least_time']
            num_spks = config['RIR_mixing']['num_spks']
            num_ch = config['RIR_mixing']['num_ch']
            scp_list = config['RIR_mixing']['scp_list']
            if config['RIR_mixing']['select_mode'] == 1:  # save train 
                tr_wave_list = config['RIR_mixing']['tr_parent_wave_list']; save_tr_pickle_dir = config['RIR_mixing']['saved_tr_pickle_dir']
                main_rirmixing('Train',num_ch, fs, chunk_time, least_time,num_spks,scp_list, tr_wave_list,save_tr_pickle_dir)  

        if args.dataset == 'SMS_WSJ':
            from dataloader.SMS_WSJ import main_smswsj
            rootDir = config['SMS_WSJ']['rootdir']; cleanDir = config['SMS_WSJ']['clean']; mixDir = config['SMS_WSJ']['mix']
            earlyDir = config['SMS_WSJ']['early']; tailDir = config['SMS_WSJ']['tail']; noiseDir = config['SMS_WSJ']['noise']
            trFile = config['SMS_WSJ']['tr_file']; devFile = config['SMS_WSJ']['dev_file']; testFile = config['SMS_WSJ']['test_file'] 
            saverootDir = config['SMS_WSJ']['saverootdir']
            main_smswsj(num_spks, num_ch, chunk_time, least_time, fs, rootDir, saverootDir, cleanDir, mixDir,earlyDir, tailDir, noiseDir, trFile, devFile, testFile)
            

    if args.mode == 'Train':
        from dataloader.data import AudioDataset
        tr_pickle_dir = config[args.dataset]['saved_tr_pickle_dir']
        dt_pickle_dir = config[args.dataset]['saved_dt_pickle_dir']
        
        # models
        if args.train_mode == 'MISO1':
            model = MISO_1(num_spks,num_ch,**config['MISO_1'])
            if args.use_cuda:
                model = model.cuda(config['gpu_num'])
            print('-'*85)
            print('-'*30, 'MOSO_1', '-'*30)
            print(model)
            print('-'*85)    

        elif args.train_mode == 'Beamforming':
            model = MISO_1(num_spks,num_ch,**config['MISO_1'])
            if args.use_cuda:
                model = model.cuda(config['gpu_num'])
            print('-'*85)
            print('-'*30, 'MOSO_1', '-'*30)
            print(model)
            print('-'*85)   
            
        elif args.train_mode == 'MISO2':
            model_sep = MISO_1(num_spks,num_ch,**config['MISO_1'])
            model = MISO_2(num_spks, num_ch, **config['MISO_2'])
            
            if args.use_cuda:
                model_sep = model_sep.cuda(config['gpu_num'])
                model = model.cuda(config['gpu_num'])

            if not config['trainer_en']['load_MISO1_Output']:
                # load pretrained MISO1 model
                MISO1_path = config['trainer_en']['MISO1_path']
                package = torch.load(MISO1_path, map_location="cuda"+str(config['gpu_num']))
                model_sep.load_state_dict(package['model_state_dict'])
                model_sep.eval()
                print('-'*85)
                print('Loading MISO1 model %s'% MISO1_path)
                print('-'*85)

            print('-'*85)
            print('-'*30, 'MOSO_1', '-'*30)
            print(model_sep)
            print('-'*85)
            print('-'*85)
            print('-'*30, 'MOSO_2', '-'*30)
            print(model)
            print('-'*85)

        elif args.train_mode == 'MISO3':
            model_sep = MISO_1(num_spks,num_ch,**config['MISO_1'])
            model = MISO_3(1, num_ch,**config['MISO_3'])
            
            if args.use_cuda:
                model_sep = model_sep.cuda(config['gpu_num'])
                model = model.cuda(config['gpu_num'])

            if not config['trainer_en']['load_MISO1_Output']:
                # load pretrained MISO1 model
                MISO1_path = config['trainer_en']['MISO1_path']
                package = torch.load(MISO1_path, map_location="cuda:"+str(config['gpu_num']))
                model_sep.load_state_dict(package['model_state_dict'])
                model_sep.eval()
                print('-'*85)
                print('Loading MISO1 model %s'% MISO1_path)
                print('-'*85)

            print('-'*85)
            print('-'*30, 'MOSO_1', '-'*30)
            print(model_sep)
            print('-'*85)
            print('-'*85)
            print('-'*30, 'MOSO_3', '-'*30)
            print(model)
            print('-'*85)


        # dataloader
        if args.train_mode == 'Separate':
            functionMode = 'Separate'
            tr_dataset = AudioDataset('Train',functionMode,num_spks,num_ch,tr_pickle_dir,ref_ch=config[args.dataset]['ref_ch'],model=None,device=None,cudaUse=False,check_audio=config['dataloader']['check_audio'],dereverb_Info=config['dereverb_Info'],**config['STFT'])
            dt_dataset = AudioDataset('Test',functionMode,num_spks,num_ch,dt_pickle_dir,ref_ch=config[args.dataset]['ref_ch'],mode=None,device=None,cudaUse=False,check_audio=config['dataloader']['check_audio'],dereverb_Info=config['dereverb_Info'],**config['STFT'])
        
        elif args.train_mode == 'Beamforming':
            functionMode = 'Beamforming'
            tr_dataset = AudioDataset('Train',functionMode,num_spks,num_ch,tr_pickle_dir,ref_ch=config[args.dataset]['ref_ch'],model=None,device=None,cudaUse=False,check_audio=config['dataloader']['check_audio'],dereverb_Info=config['dereverb_Info'],**config['STFT'])
            dt_dataset = AudioDataset('Test',functionMode,num_spks,num_ch,dt_pickle_dir,ref_ch=config[args.dataset]['ref_ch'],model=None,device=None,cudaUse=False,check_audio=config['dataloader']['check_audio'],dereverb_Info=config['dereverb_Info'],**config['STFT'])
        
        elif (args.train_mode == 'MISO2') or (args.train_mode == 'MISO3'):
            if config['trainer_en']['load_MISO1_Output'] and config['trainer_en']['load_MVDR_Output']:
                functionMode = 'Enhance_Load_MISO1_MVDR_Output'
                tr_dataset = AudioDataset('Train',functionMode,num_spks,num_ch,tr_pickle_dir,ref_ch=config[args.dataset]['ref_ch'],model=None,device=None,cudaUse=False,check_audio=config['dataloader']['check_audio'],dereverb_Info=config['dereverb_Info'],**config['STFT'])
                dt_dataset = AudioDataset('Test',functionMode,num_spks,num_ch,dt_pickle_dir,ref_ch=config[args.dataset]['ref_ch'],model=None,device=None,cudaUse=False,check_audio=config['dataloader']['check_audio'],dereverb_Info=config['dereverb_Info'],**config['STFT'])
            elif config['trainer_en']['load_MISO1_Output'] :
                functionMode = 'Enhance_Load_MISO1_Output'
                tr_dataset = AudioDataset('Train',functionMode,num_spks,num_ch,tr_pickle_dir,ref_ch=config[args.dataset]['ref_ch'],model=None,device=None,cudaUse=False,check_audio=config['dataloader']['check_audio'],dereverb_Info=config['dereverb_Info'],**config['STFT'])
                dt_dataset = AudioDataset('Test',functionMode,num_spks,num_ch,dt_pickle_dir,ref_ch=config[args.dataset]['ref_ch'],model=None,device=None,cudaUse=False,check_audio=config['dataloader']['check_audio'],dereverb_Info=config['dereverb_Info'],**config['STFT'])
            elif config['trainer_en']['load_MVDR_Output']:
                functionMode = 'Enhance_Load_MVDR_Output'
                tr_dataset = AudioDataset('Train',functionMode,num_spks,num_ch,tr_pickle_dir,ref_ch=config[args.dataset]['ref_ch'],model=model_sep,device=config['gpu_num'],cudaUse=args.use_cuda,check_audio=config['dataloader']['check_audio'],dereverb_Info=config['dereverb_Info'],**config['STFT'])
                dt_dataset = AudioDataset('Test',functionMode,num_spks,num_ch,dt_pickle_dir,ref_ch=config[args.dataset]['ref_ch'],model=model_sep,device=config['gpu_num'],cudaUse=args.use_cuda,check_audio=config['dataloader']['check_audio'],dereverb_Info=config['dereverb_Info'],**config['STFT'])    
            else:
                functionMode = 'Enhance_Scratch'
                tr_dataset = AudioDataset('Train',functionMode,num_spks,num_ch,tr_pickle_dir,ref_ch=config[args.dataset]['ref_ch'],model=model_sep,device=config['gpu_num'],cudaUse=args.use_cuda,check_audio=config['dataloader']['check_audio'],dereverb_Info=config['dereverb_Info'],**config['STFT'])
                dt_dataset = AudioDataset('Test',functionMode,num_spks,num_ch,dt_pickle_dir,ref_ch=config[args.dataset]['ref_ch'],model=model_sep,device=config['gpu_num'],cudaUse=args.use_cuda,check_audio=config['dataloader']['check_audio'],dereverb_Info=config['dereverb_Info'],**config['STFT'])   
        else:
            assert -1, '[Error] Choose correct train mode'
            
        tr_loader = DataLoader(tr_dataset, **config['dataloader']['Train'])
        dt_loader = DataLoader(dt_dataset,**config['dataloader']['Development'])
        
        # optimizer
        if config['optimizer']['name'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr = config['optimizer']['lr'],
                                         weight_decay = config['optimizer']['weight_decay'])
        if config['scheduler']['name'] == 'plateau':
            factor = config['scheduler']['factor']
            patience = config['scheduler']['patience']
            min_lr = config['scheduler']['min_lr']
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=factor,patience=patience,min_lr=min_lr)

        #trainer
        if args.train_mode == 'MISO1':
            trainer = Trainer_Separate(args.dataset, num_spks,tr_loader, dt_loader, model,optimizer,scheduler,config,config['gpu_num'], args.log_path)
        elif args.train_mode == 'Beamforming':
            trainer = Trainer_Beamforming(args.dataset, num_spks, tr_loader, dt_loader, model,config,config['gpu_num'],args.log_path) 
        elif (args.train_mode == 'MISO2') or (args.train_mode == 'MISO3'): 
            # Checking
            trainer = Trainer_Enhance(args.dataset, args.train_mode, num_spks, tr_loader, tr_loader, model,optimizer,scheduler,config,config['gpu_num'], args.log_path)
            
        trainer.train()
    
    # if args.mode == 'Test_MISO_1':
    #     from dataloader.data import AudioDataset
    #     tr_pickle_dir = config[args.dataset]['saved_tr_pickle_dir']
    #     dt_pickle_dir = config[args.dataset]['saved_dt_pickle_dir']
    #     tr_dataset = AudioDataset('Train',tr_pickle_dir,**config['STFT'])
    #     tr_loader = DataLoader(tr_dataset, **config['dataloader']['TestMISO_1'])
    #     model_sep = MISO_1(num_spks,num_ch,**config['MISO_1']).cuda(config['gpu_num'],num_spks, num_ch)
    #     print(model_sep)
    #     tester = Tester(tr_loader,model_sep,config,config['gpu_num'])
    #     tester.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type = str, default = '.', help = 'Root directory')
    parser.add_argument('-c', '--config', type = str, help = 'Path to option TAML file.')
    parser.add_argument('-d', '--dataset', type = str, help = 'Dataset')
    parser.add_argument('-m', '--mode', type = str, help= 'Extract or Train or Test')
    parser.add_argument('-u', '--use_cuda', type = int, default=1, help='Whether use GPU')
    parser.add_argument('-n', '--log_path', type = str, default='./runs/', help='tensorboard log path')
    parser.add_argument('-t', '--train_mode', type= str, default= 'MISO1', help='choose the mode to train')
    
    args = parser.parse_args()
    # with open(os.path.join(args.config, args.dataset + '.yml'), mode = 'r') as f:
    with open(os.path.join(args.config, 'NN_BSS.yml'),mode='r') as f:
        config = yaml.load(f,Loader = yaml.FullLoader)
    run(args,config)
    