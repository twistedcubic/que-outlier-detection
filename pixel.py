
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import numpy as np
import numpy.linalg as linalg
import sklearn.decomposition as decom
from scipy.stats import ortho_group
import scipy.stats as st
import random
import utils
import os.path as osp
import data
import baselines
import words
import mean
import cifar_corruptor as cif
import pdb

'''
Outlier detection and mean estimation on CIFAR data.
'''

def test_pixel_dirs(opt):
    opt.n = 8000 #10000 #50
    opt.feat_dim = 1024 #1400 #1000
    n = opt.n
    feat_dim = opt.feat_dim
    #number of top dirs for calculating tau0
    opt.n_top_dir = 1
    #number of directions to add noise
    opt.p = 0.2 #default total portion corrupted

    #use original samples for whitening 
    same_whitening_samples = False
    cif_data = cif.init()
    #number of directions
    n_dir_l = list(range(1, 16, 3))
    n_repeat = 20 
    data_l = []
    n_sample = 5000
    for n_dir in n_dir_l:
        cur_data_l = []        
        for _ in range(n_repeat):
            if same_whitening_samples:
                sample_idx = np.random.randint(low=0, high=n_sample, size=(n_sample,))
            else:
                sample_idx = None
            if opt.whiten:                
                whiten_mx = cif.get_whitening(cif_data, fast_whiten=opt.fast_whiten, sample_idx=sample_idx)
            else:
                whiten_mx = np.eye(feat_dim)
            X, X_n = cif.get_corrupted_data(cif_data, n_dir, opt.p, whiten_mx, fast_whiten=opt.fast_whiten, sample_idx=sample_idx)
            X, X_n = torch.from_numpy(X.astype(np.float32)).to(utils.device), torch.from_numpy(X_n).to(utils.device, torch.float32)
            noise_idx = torch.LongTensor(list(range(len(X_n)))).to(utils.device) + len(X)
            X = torch.cat((X, X_n), dim=0)
            #pdb.set_trace()
            X = X - X.mean(0)
            if opt.fast_jl:
                ##enable if doing fast JL
                X = utils.pad_to_2power(X)
            cur_data_l.append((X, noise_idx))
        data_l.append(cur_data_l)
    
    print('samples feat dim {}'.format(X.size(1)))
        
    #which baseline to use as tau0, can be 'isolation_forest'
    opt.baseline = 'tau0' #'l2' #'tau0' #'isolation_forest' #'l2' #
    print('baseline method: {}'.format(opt.baseline))    
    opt.n_iter = 1
    #amount to remove wrt cur_p
    opt.remove_factor = 1./opt.n_iter    
    
    #scalar to multiply norm of noise vectors with. This is deprecated
    opt.norm_scale = 1.3
    #amount to divide noise norm by
    opt.noise_norm_div = 8
    opt.lamb_multiplier = 6
    
    #opt.n_dir = N_DIR
    #n_dir = opt.n_dir
    acc_l = []
    #numpy array used for plotting.
    k_l = []
    p_l = []
    tau_l = []
    res_l = []
    
    #no need to include tau0
    if opt.fast_jl:
        outlier_methods_l = ['l2'] 
    else:
        outlier_methods_l = ['l2', 'iso forest', 'ell env', 'lof', 'knn']

    #+3 for tau1 tau0 and n_dir
    scores_ar = np.zeros((len(n_dir_l), len(outlier_methods_l)+3))
    std_ar = np.zeros((len(n_dir_l), len(outlier_methods_l)+3))
    
    for j, n_dir in enumerate(n_dir_l):
        
        cur_data_l = data_l[j]
        opt.n_dir = n_dir
        
        #percentage to remove   
        opt.remove_p = opt.p*opt.remove_factor
        #for cur_dir in range(3, n_dir, 9):            
        #cur_res_l = [n, feat_dim, n_noise_dir, opt.p, opt.lamb_multiplier, opt.norm_scale]
        acc_mx = torch.zeros(n_repeat, 2)
        cur_scores_ar = np.zeros((n_repeat, len(outlier_methods_l)+2))
        for i in range(n_repeat):
            X, noise_idx = cur_data_l[i]
            ##cur_scores_ar[i] = train(X, n_noise_dir, opt.p, outlier_methods_l, opt)            
            cur_scores_ar[i] = test_pixel2(X, noise_idx, outlier_methods_l, opt)
            acc_mx[i, 0] = cur_scores_ar[i, 1] #acc0
            acc_mx[i, 1] = cur_scores_ar[i, 0] #acc1
            
        scores_ar[j, 1:] = np.mean(cur_scores_ar, axis=0)
        if opt.use_std:
            std_ar[j, 1:] = np.std(cur_scores_ar, axis=0)
        else:
            se = np.clip(st.sem(cur_scores_ar, axis=0), 1e-3, None)
            low, high = st.t.interval(0.95, cur_scores_ar.shape[0]-1, loc=scores_ar[j, 1:], scale=se)
            std_ar[j, 1:] = (high - low)/2.
        
        scores_ar[j, 0] = n_dir
        std_ar[j, 0] = n_dir
        
        acc_mean = acc_mx.mean(dim=0)
        acc0, acc1 = acc_mean[0].item(), acc_mean[1].item()
        print('n_noise_dir {} lamb {} acc0 {} acc1 {}'.format(n_dir, opt.lamb_multiplier, acc0, acc1))
        #cur_res_l.extend([acc0, acc1])
        
    print('About to plot!')
    print(std_ar)
    pdb.set_trace()
    #if plot_lambda:
    #legends = ['lamb', 'acc', 'tau']
    #else:
    #    legends = ['k', 'acc', 'tau', 'p']
    #plot both tau1 vs tau0, and tau1 against all baselines.
    ##utils.plot_acc_syn_lamb(p_l, acc_l, tau_l, legends, opt)
    
    scores_ar = scores_ar.transpose()
    std_ar = std_ar.transpose()
    utils.plot_scatter_flex(scores_ar, ['tau1', 'tau0'] + outlier_methods_l, opt, std_ar=std_ar)
    m = {'opt':opt, 'scores_ar':scores_ar, 'conf_ar':std_ar}
    with open(osp.join('results', opt.dir, 'dirs_data.npy'), 'wb') as f:
        torch.save(m, f)
        print('saved under {}'.format(f))

def test_pixel_lamb(opt):
    
    n_dir_l = [3, 6, 10]
    #n_dir_l = [3]
    legend_l = []
    scores_l = []
    conf_l = []
    cif_data = cif.init()
    
    if opt.compute_scores_diff:
        for n_dir in n_dir_l:
            legend_l.append(str(n_dir))
            opt.n_dir = n_dir
            mean1, conf1 = test_pixel_lamb2(cif_data, opt)
            #scores_l.append(mean1[:, 1])                                                                          
            #conf_l.append(conf1[:, 1])                                                                         
            scores_l.append(mean1)
            conf_l.append(conf1)

        n_lamb = mean1.shape[-1]
        scores_ar = np.stack(scores_l, axis=0)
        conf_ar = np.stack(conf_l, axis=0)
        tau0_ar = np.concatenate((mean1[0].reshape(1, -1), scores_ar[:,2,:].reshape(len(n_dir_l), n_lamb)), axis=0)
        tau0_conf_ar = np.concatenate((mean1[0].reshape(1, -1), conf_ar[:,2,:].reshape(len(n_dir_l), n_lamb)), axis=0)

        l2_ar = np.concatenate((mean1[0].reshape(1, -1), scores_ar[:,3,:].reshape(len(n_dir_l), n_lamb)), axis=0)
        l2_conf_ar = np.concatenate((mean1[0].reshape(1, -1), conf_ar[:,3,:].reshape(len(n_dir_l), n_lamb)), axis=0)
        #scores_ar = np.concatenate((mean1[:, 0].reshape(1, -1), scores_ar[:,:,3]), axis=0)
        #scores_ar = np.stack([mean1[:, 0]]+conf_l, axis=0)
        #np.concatenate((mean1[:, 0].reshape(1,-1), np.stack(scores_l, axis=0)), axis=0) 
        pdb.set_trace()
        utils.plot_scatter_flex(tau0_ar, legend_l, opt, std_ar=tau0_conf_ar, name='tau0')
        utils.plot_scatter_flex(l2_ar, legend_l, opt, std_ar=l2_conf_ar, name='l2')
    else:
        for n_dir in n_dir_l:
            legend_l.append(str(n_dir))
            opt.n_dir = n_dir
            mean1, conf1 = test_pixel_lamb2(cif_data, opt)
            scores_l.append(mean1[1])
            conf_l.append(conf1[1])

        scores_ar = np.stack([mean1[0]]+scores_l, axis=0)
        conf_ar = np.stack([mean1[0]]+conf_l, axis=0)
        #scores_ar = np.concatenate((mean1[:, 0].reshape(1,-1), np.stack(scores_l, axis=0)), axis=0)
        #conf_ar = np.concatenate((mean1[:, 0].reshape(1,-1), np.stack(conf_l, axis=0)), axis=0)
        pdb.set_trace()
        utils.plot_scatter_flex(scores_ar, legend_l, opt, std_ar=conf_ar)    
        
    m = {'opt':opt, 'scores_ar':scores_ar, 'conf_ar':conf_ar}
    with open(osp.join('results', opt.dir, 'lamb_data.npy'), 'wb') as f:
        torch.save(m, f)
        print('saved under {}'.format(f))

'''
Returns:
-mean and confidence intervals of various scores, tau1 + baselines.
'''
def test_pixel_lamb2(cif_data, opt):

    #number of top dirs for calculating tau0
    opt.n_top_dir = 1
    #number of directions to add noise
    opt.p = 0.2 #default total portion corrupted
    
    n_repeat = 5
    data_l = []
    
    for _ in range(n_repeat):
        whiten_mx = cif.get_whitening(cif_data, fast_whiten=opt.fast_whiten)
        X, X_n = cif.get_corrupted_data(cif_data, opt.n_dir, opt.p, whiten_mx, fast_whiten=opt.fast_whiten)
        X, X_n = torch.from_numpy(X.astype(np.float32)).to(utils.device), torch.from_numpy(X_n).to(utils.device, torch.float32)
        noise_idx = torch.LongTensor(list(range(len(X_n)))).to(utils.device) + len(X)
        X = torch.cat((X, X_n), dim=0)        
        X = X - X.mean(0)
        
        if opt.fast_jl:
            X = utils.pad_to_2power(X)
        data_l.append((X, noise_idx))
    print('samples feat dim: {}'.format(X.size(1)))
        
    #which baseline to use as tau0, can be 'isolation_forest'
    opt.baseline = 'tau0' #'l2' #'tau0' #'isolation_forest' #'l2' #
    print('baseline method: {}'.format(opt.baseline))    
    opt.n_iter = 1
    #amount to remove wrt cur_p
    opt.remove_factor = 1./opt.n_iter    
    
    #scalar to multiply norm of noise vectors with. This is deprecated
    opt.norm_scale = 1.3
    #amount to divide noise norm by
    opt.noise_norm_div = 8
    
    #opt.n_dir = N_DIR
    #n_dir = opt.n_dir
    acc_l = []
    #numpy array used for plotting.
    k_l = []
    p_l = []
    tau_l = []
    res_l = []
    
    #no need to include tau0
    if opt.fast_whiten:
        #for studying lambda only need to compare with best baselines
        outlier_methods_l = ['l2']    
    else:    
        outlier_methods_l = ['l2', 'iso forest', 'ell env', 'lof', 'knn']
    
    lamb_l = list(range(0, 22, 3))
    #+3 for tau1 tau0 and lamb
    scores_ar = np.zeros((len(lamb_l), len(outlier_methods_l)+3))
    std_ar = np.zeros((len(lamb_l), len(outlier_methods_l)+3))
    
    for j, lamb in enumerate(lamb_l):
        
        opt.lamb_multiplier = lamb                    
        #percentage to remove   
        opt.remove_p = opt.p*opt.remove_factor
        #for cur_dir in range(3, n_dir, 9):            
        #cur_res_l = [n, feat_dim, n_noise_dir, opt.p, opt.lamb_multiplier, opt.norm_scale]
        acc_mx = torch.zeros(n_repeat, 2)
        cur_scores_ar = np.zeros((n_repeat, len(outlier_methods_l)+2))
        for i in range(n_repeat):
            X, noise_idx = data_l[i]
            ##cur_scores_ar[i] = train(X, n_noise_dir, opt.p, outlier_methods_l, opt)            
            cur_scores_ar[i] = test_pixel2(X, noise_idx, outlier_methods_l, opt)
            acc_mx[i, 0] = cur_scores_ar[i, 1] #acc0
            acc_mx[i, 1] = cur_scores_ar[i, 0] #acc1
            
        '''
        if opt.use_std:
            std_ar[j, 1:] = np.std(cur_scores_ar, axis=0)
        else:
            se = np.clip(st.sem(cur_scores_ar, axis=0), 1e-3, None)        
            low, high = st.t.interval(0.95, cur_scores_ar.shape[0]-1, loc=scores_ar[j, 1:], scale=se)
            std_ar[j, 1:] = (high - low)/2.
        '''
        
        if opt.compute_scores_diff:
            #tau1 - tau0
            cur_scores_ar[:, 1] = cur_scores_ar[:, 0] - cur_scores_ar[:, 1]
            cur_scores_ar[:, 2] = cur_scores_ar[:, 0] - cur_scores_ar[:, 2]
            
        scores_ar[j, 1:] = cur_scores_ar.mean(axis=0)        
        if opt.use_std:
            std_ar[j, 1:] = cur_scores_ar.std(axis=0)            
        else:
            #low, high = st.t.interval(0.95, n_repeat-1, loc=auc_prob1, scale=st.sem(cur_auc_prob1_l))
            #conf_int1 = (high - low)/2.
            se = np.clip(st.sem(cur_scores_ar, axis=0), 1e-4, None) 
            low, high = st.t.interval(0.95, n_repeat-1, loc=scores_ar[j, 1:], scale=se)
            std_ar[j, 1:] = (high - low)/2.
        
        scores_ar[j, 0] = lamb
        
    scores_ar = scores_ar.transpose()
    std_ar = std_ar.transpose()
    print(std_ar)
    plot = False
    if plot:
        print('About to plot!')
        pdb.set_trace()
        utils.plot_scatter_flex(scores_ar, ['tau1', 'tau0'] + outlier_methods_l, opt, std_ar=std_ar)
        
    return scores_ar, std_ar
    
'''
Returns:
-scores of tau1 and baselines, length of outlier_method_l + 2
'''
def test_pixel2(X, noise_idx, outlier_method_l, opt):
    
    '''
    content_path = 'data/sherlock.txt' if content_lines is None else None
    #noise_path = 'data/news_noise1.txt' if noise_lines is not None else None
    noise_path = 'data/sherlock_noise3.txt' if noise_lines is None else None
    '''
    #words_ar, X, noise_idx = words.doc_word_embed_content_noise(content_path, noise_path, 'data/sherlock_whiten.txt', content_lines, noise_lines)#.to(utils.device) #('data/sherlock_noise3.txt', 'data/test_noise.txt')#.to(utils.device)
    
    noise_idx = noise_idx.unsqueeze(-1)
    print('** {} number of outliers {}'.format(X.size(0), len(noise_idx)))
    #pdb.set_trace()
    
    opt.n, opt.feat_dim = X.size(0), X.size(1)
    #percentage of points to remove.
    opt.remove_p = 0.2
    #number of top dirs for calculating tau0.
    opt.n_top_dir = 1
    opt.n_iter = 1 
    #use select_idx rather than the scores tau, since tau's are scores for remaining points after outliers.
    tau1, select_idx1, n_removed1, tau0, select_idx0, n_removed0 = mean.compute_tau1_tau0(X, opt)
    ##tau1, select_idx1, n_removed1, tau0, select_idx0, n_removed0 = torch.ones(len(X)).to(utils.device), None, 5, torch.ones(len(X)).to(utils.device), None, 5 #mean.compute_tau1_tau0(X, opt)
    
    all_idx = torch.zeros(X.size(0), device=utils.device)
    ones = torch.ones(noise_idx.size(0), device=utils.device)
    
    all_idx.scatter_add_(dim=0, index=noise_idx.squeeze(), src=ones)
    
    opt.baseline = 'tau0' #'lof'#'knn' 'l2' #'l2' #'tau0' #'l2'#'isolation_forest'#'dbscan' #'isolation_forest'
    scores_l = []
    
    for method in outlier_method_l:
        if method == 'iso forest':
            tau = baselines.isolation_forest(X)
        elif method == 'ell env':
            tau = baselines.ellenv(X)
        elif method == 'lof':
            tau = baselines.knn_dist_lof(X)
        elif method == 'dbscan':
            tau = baselines.dbscan(X)
        elif method == 'l2':
            tau = baselines.l2(X)        
        elif method == 'knn':
            tau = baselines.knn_dist(X)
        elif method == 'tau2':
            select_idx2 = torch.LongTensor(list(range(len(X)))).to(utils.device)
            tau = mean.compute_tau2(X, select_idx2, opt)        
        else:
            raise Exception('Outlier method {} not supported'.format(method))
        good_scores = tau[all_idx==0]
        bad_scores = tau[all_idx==1]
        auc = utils.auc(good_scores, bad_scores)
        scores_l.append(auc)
        
    if opt.n_iter > 1:
        #all_idx = torch.LongTensor(range(len(X_classes))).to(utils.device)
        all_idx = torch.LongTensor(range(len(X))).to(utils.device)
        zeros1 = torch.zeros(len(X), device=utils.device)
        zeros1[select_idx1] = 1
        outliers_idx1 = all_idx[zeros1==0]
        zeros0 = torch.zeros(len(X), device=utils.device)
        zeros0[select_idx0] = 1
        outliers_idx0 = all_idx[zeros0==0]
        if opt.baseline != 'tau0': 
            outliers_idx0 = torch.topk(tau0, k=n_removed0, largest=True)[1]
            
    else:
        #should not be used if n_iter > 1
        outliers_idx0 = torch.topk(tau0, k=n_removed0, largest=True)[1]
        outliers_idx1 = torch.topk(tau1, k=n_removed1, largest=True)[1]
        
        #Distribution of true outliers with respect to the predicted scores.
        compute_auc_b = True
        if compute_auc_b:
            #complement of noise_idx
            #X_range = list(range(len(X)))
            zeros = torch.zeros(len(tau1), device=utils.device)
            zeros[noise_idx] = 1
            
            inliers_tau1 = tau1[zeros==0] #this vs index_select
            outliers_tau1 = tau1[zeros==1]#torch.index_select(tau1, dim=0, index=noise_idx)
            ##utils.inlier_outlier_hist(inliers_tau1, outliers_tau1, 'tau1', high=40)
            tau1_auc = utils.auc(inliers_tau1, outliers_tau1)
            
            inliers_tau0 = tau0[zeros==0] #this vs index_select
            outliers_tau0 = tau0[zeros==1] #torch.index_select(tau0, dim=0, index=noise_idx)
            ##utils.inlier_outlier_hist(inliers_tau0, outliers_tau0, opt.baseline, high=40)            
            tau0_auc = utils.auc(inliers_tau0, outliers_tau0)
            
    print('tau1 size {}'.format(tau1.size(0)))
    outliers_idx0_exp = outliers_idx0.unsqueeze(0).expand(len(noise_idx), -1)
    outliers_idx1_exp = outliers_idx1.unsqueeze(0).expand(len(noise_idx), -1)
    assert len(outliers_idx0) == len(outliers_idx1)
    
    tau0_cor = noise_idx.eq(outliers_idx0_exp).sum()
    tau1_cor = noise_idx.eq(outliers_idx1_exp).sum()
    print('{}_cor {} out of {} tau1_cor {} out of {}'.format(opt.baseline, tau0_cor, len(outliers_idx0), tau1_cor, len(outliers_idx1)))    
    
    #return tau0_cor.item()/len(outliers_idx0), tau1_cor.item()/len(outliers_idx0), tau0_auc, tau1_auc #0 instead of 1
    return [tau1_auc, tau0_auc] + scores_l


if __name__ == '__main__':
    opt = utils.parse_args()
    
    opt.use_std = True
    opt.compute_scores_diff = True
    opt.whiten = True
    opt.fast_whiten = True
    
    #directory to store results
    opt.dir = 'cifar'
    method = opt.experiment_type
    if method == 'image_lamb':
        opt.type = 'lamb'
        test_pixel_lamb(opt)
    elif method == 'image_dirs':
        opt.type = 'dirs'
        test_pixel_dirs(opt)
    else:
        raise Exception('Wrong script for experiment type {}'.format(method))
    

