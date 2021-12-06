import numpy as np
from numpy.linalg import solve, det
import pdb
import torch

def apply_wpe(observe_stft,tapDelay,nTap):
    B, F, M, T = observe_stft.size()

    return 

def apply_dnn_wpe(observe_stft,source_stft,tapDelay,nTap,reverb_variance_flowValue):
    """
    Input:
        observe_stft, source_stft : [B, F, Ch, T]
    """
    
    B,F,M,T = observe_stft.shape
    observe_stft_pad = np.concatenate((np.zeros([B,F,M,nTap+tapDelay-1],dtype=np.complex),observe_stft), axis=3) # [B,F,M,Pad+T]
    nFramePad = T + nTap + tapDelay -1
    dereverb_stft = np.zeros([B,F,M,T],dtype=np.complex)
    

    for b_idx in range(B):
        # Initialize
        LambdaInv = np.zeros([M*nTap,F,M,T]) #[M*nTap, F,M,T]
        Lambda = np.zeros([F,M,T])

        # Lambda =  np.maximum( abs(source_stft[b_idx,...])**2, 1e-8)  #[F,M,T]
        # for sensorTap in range(0,M*nTap):
        #     LambdaInv[sensorTap,...] = 1/(Lambda+1e-6)

        for ch_idx in range(M):
            Lambda[:,ch_idx,:]= np.maximum(np.max(reverb_variance_flowValue * abs(source_stft[b_idx,:,ch_idx,:])**2), abs(source_stft[b_idx,:,ch_idx,:])**2)
        for sensorTap in range(0,M*nTap):
            LambdaInv[sensorTap,...] = 1/(Lambda)

        XDelay = np.zeros([M*nTap,T],dtype=np.complex) # [M*nTap, T]
        for ch_idx in range(M):
            for f_idx in range(F):
                Xb = observe_stft_pad[b_idx,f_idx,...].T  # [Pad+T,M]
                for t_idx in range(nTap+tapDelay,nFramePad):
                    # XDelay[:,t_idx-nTap-tapDelay] = np.reshape(np.flip(Xb[t_idx-nTap-tapDelay : t_idx-tapDelay,:],axis=0),[1,-1]) 
                    XDelay[:,t_idx-nTap-tapDelay] = np.reshape(Xb[t_idx-nTap-tapDelay : t_idx-tapDelay,:],[-1]) 

                temp = np.conj(XDelay) * LambdaInv[:,f_idx,ch_idx, :] #[M*nTap, T]
                R = np.einsum('...dt,...et->...de', temp , XDelay) / T  # [M*nTap, T] * [T, M*nTap] = [M*nTap, M*nTap]
                r = np.einsum('...dt,...et->...de',temp , np.expand_dims(observe_stft[b_idx,f_idx,ch_idx,:],axis=0)) / T # [M*nTap, T] * [T,1] = [M*nTap,1]

                # if det(R) == 0:
                #     print('Warning Determinant is 0, Bypass')
                #     g = np.zeros([M*nTap,1])
                # else:
                #     g = solve(R,r) # [M*nTap, 1]
                
                g = solve(R,r) # [M*nTap, 1]
                
                dereverb_stft[b_idx,f_idx,ch_idx,:] = observe_stft[b_idx,f_idx,ch_idx,:] - np.squeeze(np.einsum('...et,...dt->...ed', XDelay.T, g.T)) # [T,M*nTap] * [M*ntap,1] = [T,1]
               
    dereverb_stft = torch.permute(torch.from_numpy(dereverb_stft), [0,2,3,1])

    return dereverb_stft
                

def apply_fcp(observe_stft, source_stft,nTap,reverb_variance_flowValue):  
    """
    Input:
        observe_stft, source_stft : [B, F, Ch, T]
    """
    B,F,M,T = observe_stft.shape
    nFramePad = T + nTap
    source_stft_pad = np.concatenate((np.zeros([B,F,M,nTap-1],dtype=np.complex),source_stft), axis=3) # [B,F,M,Pad+T-1]
    dereverb_stft = np.zeros([B,F,M,T],dtype=np.complex)

    for b_idx in range(B):
        # Initialize
        LambdaInv = np.zeros([M*nTap,F,M,T]) #[M*nTap, F,M,T]
        Lambda = np.zeros([F,M,T])

        # Lambda =  np.maximum( abs(source_stft[b_idx,...])**2, epsi)  #[F,M,T]
        # for sensorTap in range(0,M*nTap):
        #     LambdaInv[sensorTap,...] = 1/(Lambda+epsi)

        for ch_idx in range(M):
            # Lambda[:,ch_idx,:]= np.maximum(np.max(reverb_variance_flowValue * abs(source_stft[b_idx,:,ch_idx,:])**2), abs(source_stft[b_idx,:,ch_idx,:])**2)
            Lambda[:,ch_idx,:]= np.maximum(np.max(reverb_variance_flowValue * abs(observe_stft[b_idx,:,ch_idx,:])**2), abs(observe_stft[b_idx,:,ch_idx,:])**2)
        for sensorTap in range(0,M*nTap):
            LambdaInv[sensorTap,...] = 1/(Lambda)

        SDelay = np.zeros([M*nTap,T],dtype=np.complex) # [M*nTap, T]

        for ch_idx in range(M):
            for f_idx in range(F):
                Xb = source_stft_pad[b_idx,f_idx,...].T  # [Pad+T-1,M]
                for t_idx in range(nTap,nFramePad):
                    SDelay[:,t_idx-nTap] = np.reshape(np.flip(Xb[t_idx-nTap : t_idx,:],axis=0),[-1]) 
                    # SDelay[:,t_idx-nTap] = np.reshape(Xb[t_idx-nTap : t_idx,:],[-1]) 

                    
                temp = np.conj(SDelay) * LambdaInv[:,f_idx,ch_idx, :] #[M*nTap, T]
                R = np.einsum('...dt,...et->...de', temp , SDelay) / T  # [M*nTap, T] * [T, M*nTap] = [M*nTap, M*nTap]
                r = np.einsum('...dt,...et->...de',temp , np.expand_dims(observe_stft[b_idx,f_idx,ch_idx,:],axis=0)) / T # [M*nTap, T] * [T,1] = [M*nTap,1]

                # if det(R) == 0:
                #     print('Warning Determinant is 0, Bypass')
                #     g = np.zeros([M*nTap,1])
                # else:
                #     g = solve(R,r) # [M*nTap, 1]
                
                g = solve(R,r) # [M*nTap, 1]
                
                dereverb_stft[b_idx,f_idx,ch_idx,:] = observe_stft[b_idx,f_idx,ch_idx,:] - (np.squeeze(np.einsum('...et,...dt->...ed',SDelay.T, g.T)) - source_stft[b_idx,f_idx,ch_idx,:])
               
    dereverb_stft = torch.permute(torch.from_numpy(dereverb_stft), [0,2,3,1])

    return dereverb_stft

def apply_cfcp(observe_stft, source_stft, num_spks, nTap,reverb_variance_flowValue):  
    
    # observe_stft = observe_stft.astype(np.complex)
    B,F,M,T = observe_stft.shape
    nFramePad = T + nTap
    source_stft_pad = [[] for _ in range(num_spks)]
    dereverb_stft = [[] for _ in range(num_spks)]
    for spk_idx in range(num_spks):
        source_stft_pad[spk_idx] = np.concatenate((np.zeros([B,F,M,nTap-1],dtype=np.complex),source_stft[spk_idx]), axis=3)
        dereverb_stft[spk_idx] = np.zeros([B,F,M,T],dtype=np.complex) 

    for b_idx in range(B):
        # Initialize
        LambdaInv = np.zeros([M*nTap,F,M,T]) #[M*nTap, F,M,T]
        Lambda = np.zeros([F,M,T])

        for ch_idx in range(M):
            Lambda[:,ch_idx,:]= np.maximum(np.max(reverb_variance_flowValue * abs(observe_stft[b_idx,:,ch_idx,:])**2), abs(observe_stft[b_idx,:,ch_idx,:])**2)
        for sensorTap in range(0,M*nTap):
            LambdaInv[sensorTap,...] = 1/(Lambda)

        # LambdaInv = [np.zeros([M*nTap,F,M,T],dtype=np.complex) for _ in range(num_spks)] #[M*nTap, F,M,T]
        # Lambda = [np.zeros([F,M,T],dtype=np.complex) for _ in range(num_spks)]
        # for spk_idx in range(num_spks):
        #     for ch_idx in range(M):
        #         Lambda[spk_idx][:,ch_idx,:]= np.maximum(np.max(reverb_variance_flowValue * abs(source_stft[spk_idx][b_idx,:,ch_idx,:])**2), abs(source_stft[spk_idx][b_idx,:,ch_idx,:])**2)
        #     for sensorTap in range(0,M*nTap):
        #         LambdaInv[spk_idx][sensorTap,...] = 1/(Lambda[spk_idx])

        SDelay = np.zeros([M*nTap,T],dtype=np.complex) # [M*nTap, T]
                    
        for ch_idx in range(M):
            for f_idx in range(F):
                combine_reverb = np.zeros([T], dtype= np.complex)
                for spk_idx in range(num_spks):
                    # combine_reverb = np.zeros([T], dtype= np.complex)
                    Xb = source_stft_pad[spk_idx][b_idx,f_idx,...].T # [nTap+T,M]
                    for t_idx in range(nTap,nFramePad):
                        SDelay[:,t_idx-nTap] = np.reshape(np.flip(Xb[t_idx-nTap : t_idx, :], axis=0), [-1]) #[F,M*nTap,T]
                        # SDelay[:,t_idx-nTap] = np.reshape(Xb[t_idx-nTap : t_idx, :],[-1]) #[M*nTap,T]
                    
                    temp = np.conj(SDelay) * LambdaInv[:,f_idx,ch_idx,:] #[M*nTap,T]

                    R = np.einsum('...dt,...et->...de', temp, SDelay) / T # [M*nTap, T] * [M*nTap, T] -> [ M*nTap, M*nTap]
                    # r = np.einsum('...dt,...et->...de', temp, np.expand_dims(observe_stft[b_idx,f_idx,ch_idx,:],axis=0)) / T # [M*nTap, T] * [1,T]
                    r = np.einsum('...dt,...et->...de', temp, np.expand_dims(observe_stft[b_idx,f_idx,ch_idx,:]-source_stft[spk_idx][b_idx,f_idx,ch_idx,:],axis=0)) / T # [M*nTap, T] * [1,T]

                    g = solve(R,r) #[M*nTap,1]

                    combine_reverb += np.squeeze(np.einsum('...dt,...et->...de', SDelay.T, g.T)) - source_stft[spk_idx][b_idx,f_idx,ch_idx,:] #[T]
                    
                for spk_idx in range(num_spks):
                    dereverb_stft[spk_idx][b_idx,f_idx,ch_idx,:] = observe_stft[b_idx,f_idx,ch_idx,:] - combine_reverb
                    
    for spk_idx in range(num_spks):
        dereverb_stft[spk_idx] = torch.permute(torch.from_numpy(dereverb_stft[spk_idx]), [0,2,3,1])
    
    return dereverb_stft


def apply_test(observe_stft, source_stft, num_spks, nTap,reverb_variance_flowValue):  
    
    # observe_stft = observe_stft.astype(np.complex)
    B,F,M,T = observe_stft.shape
    dereverb_stft = [[] for _ in range(num_spks)]
    for spk_idx in range(num_spks):
        dereverb_stft[spk_idx] = np.zeros([B,F,M,T],dtype=np.complex) 
        dereverb_stft[spk_idx][:,:,:,:nTap-1] = observe_stft[:,:,:,:nTap-1]

    for b_idx in range(B):
        # Initialize
        LambdaInv = np.zeros([M*nTap,F,M,T]) #[M*nTap, F,M,T]
        Lambda = np.zeros([F,M,T])

        for ch_idx in range(M):
            Lambda[:,ch_idx,:]= np.maximum(np.max(reverb_variance_flowValue * abs(observe_stft[b_idx,:,ch_idx,:])**2), abs(observe_stft[b_idx,:,ch_idx,:])**2)
        for sensorTap in range(0,M*nTap):
            LambdaInv[sensorTap,...] = 1/(Lambda)

        # LambdaInv = [np.zeros([M*nTap,F,M,T],dtype=np.complex) for _ in range(num_spks)] #[M*nTap, F,M,T]
        # Lambda = [np.zeros([F,M,T],dtype=np.complex) for _ in range(num_spks)]
        # for spk_idx in range(num_spks):
        #     for ch_idx in range(M):
        #         Lambda[spk_idx][:,ch_idx,:]= np.maximum(np.max(reverb_variance_flowValue * abs(source_stft[spk_idx][b_idx,:,ch_idx,:])**2), abs(source_stft[spk_idx][b_idx,:,ch_idx,:])**2)
        #     for sensorTap in range(0,M*nTap):
        #         LambdaInv[spk_idx][sensorTap,...] = 1/(Lambda[spk_idx])

        SDelay = np.zeros([M*nTap,T-nTap],dtype=np.complex) # [M*nTap, T]
        for ch_idx in range(M):
            for f_idx in range(F):
                combine_reverb = np.zeros([T-nTap], dtype= np.complex)
                for spk_idx in range(num_spks):
                    Xb = source_stft[spk_idx][b_idx,f_idx,...].T # [nTap+T,M]
                    for t_idx in range(nTap,T):
                        SDelay[:,t_idx-nTap] = np.reshape(np.flip(Xb[t_idx-nTap : t_idx, :], axis=0), [-1]) #[M*nTap,T-nTap]
                    temp = np.conj(SDelay) * LambdaInv[:,f_idx,ch_idx,nTap:] #[M*nTap,T]

                    R = np.einsum('...dt,...et->...de', temp, SDelay) / (T-nTap) # [M*nTap, T] * [M*nTap, T] -> [ M*nTap, M*nTap]
                    r = np.einsum('...dt,...et->...de', temp, np.expand_dims(observe_stft[b_idx,f_idx,ch_idx,nTap:],axis=0)) / (T-nTap) # [M*nTap, T] * [1,T]

                    g = solve(R,r) #[M*nTap,1]

                    combine_reverb +=  np.squeeze(np.einsum('...dt,...et->...de', SDelay.T, g.T)) - source_stft[spk_idx][b_idx,f_idx,ch_idx,nTap:] #[T]
                    
                for spk_idx in range(num_spks):
                    dereverb_stft[spk_idx][b_idx,f_idx,ch_idx,nTap:] = observe_stft[b_idx,f_idx,ch_idx,nTap:] - combine_reverb

    for spk_idx in range(num_spks):
        dereverb_stft[spk_idx] = torch.permute(torch.from_numpy(dereverb_stft[spk_idx]), [0,2,3,1])
    
    return dereverb_stft


def Apply_ConvolutivePrediction(observe_stft, source_stft,num_spks, dereverb_Type, tapDelay, nTap, reverb_variance_flowValue):
    """
    Input : type : numpy.ndarray
            observe_stft : [B,F,Ch,T]
            source_stft : [[B,F,Ch,T], ...]
            dereverb_Type : WPE / DNN_WPE / ICP / FCP
    Output : type : numpy.ndarray
             Dereverb_stft : [B,Ch,T,F]
    """
    
    if dereverb_Type == 'WPE':
        dereverb_stft = apply_wpe(observe_stft,tapDelay,nTap)
    elif dereverb_Type == 'DNN_WPE':
        dereverb_stft = apply_dnn_wpe(observe_stft,source_stft,tapDelay,nTap,reverb_variance_flowValue)
    # elif dereverb_Type == 'ICP':
    #     dereverb_stft = apply_icp(observe_stft,source_stft,nTap)
    elif dereverb_Type == 'FCP':
        dereverb_stft = apply_fcp(observe_stft,source_stft,nTap,reverb_variance_flowValue)
    elif dereverb_Type == 'cFCP':
        dereverb_stft = apply_cfcp(observe_stft,source_stft,num_spks,nTap,reverb_variance_flowValue)
    elif dereverb_Type == 'test':
        dereverb_stft = apply_test(observe_stft,source_stft,num_spks,nTap,reverb_variance_flowValue)
    return dereverb_stft    