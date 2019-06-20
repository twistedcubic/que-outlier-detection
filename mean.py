
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import numpy as np
import numpy.linalg as linalg
import sklearn.decomposition as decom
from scipy.stats import ortho_group
import scipy.stats as st
import scipy as sp
import random
import utils
import os.path as osp
import data
import baselines
import words
import ads

import pdb

device = utils.device

NOISE_INN_THRESH = 0.1

#max number of directions
N_DIR = 50

DEBUG = False

'''
Compute M
'''
def compute_m(X, lamb, noise_vecs=None):
    
    X_cov = (lamb*cov(X))
    #torch svd has bug. U and V not equal up to sign or permutation, for non-duplicate entries.
    #U, D, Vt = (lamb*X_cov).svd()
    
    U, D, Vt = linalg.svd(X_cov.cpu().numpy())
    U = torch.from_numpy(U.astype('float64')).to(device)
    #torch can't take exponential on int64 types.
    D_exp = torch.from_numpy(np.exp(D.astype('float64'))).to(device).diag()
    
    #projection of noise onto the singular vecs. 
    if noise_vecs is not None:
        n_noise = noise_vecs.size(0)
        print(utils.inner_mx(noise_vecs, U)[:, :int(1.5*n_noise)])
                    
    m = torch.mm(U, D_exp)
    m = torch.mm(m, U.t())
    
    assert m.max().item() < float('Inf')    
    m_tr =  m.diag().sum()
    m = m / m_tr
    
    return m.to(torch.float32)

def compute_m0(X, lamb, noise_vecs=None):
    X_cov = (lamb*cov(X))
    u,v,w = sp.linalg.svd(X_cov.cpu().numpy())
    #pdb.set_trace()
    m = torch.from_numpy(sp.linalg.expm(lamb * X_cov.cpu().numpy() / v[0])).to(utils.device)
    m_tr =  m.diag().sum()
    m = m / m_tr
    return m

'''
Modifies in-place
(More complex choosing directions.)
@deprecated
'''
def corrupt_random_sample_dep(X, n_dir):
    
    prev_dir_l = []
    n_points = X.size(0)
    n_cor = max(1, int(n_points * cor_portion))    
    cor_idx = torch.zeros(n_dir, n_cor, dtype=torch.int64, device=X.device)
    
    for i in range(n_dir):
        cor_idx[i] = corrupt1d(X, prev_dir_l).view(-1)

    idx = torch.zeros(n_dir, n_points, device=X.device)
    src = torch.ones(1, n_cor, device=X.device).expand(n_dir, -1)
    
    idx.scatter_add_(1, cor_idx, src)
    idx = idx.sum(0)
    cor_idx = torch.LongTensor(range(n_points))[idx.view(-1)>0].to(X.device)
    
    return cor_idx

'''
Modifies in-place.
Need to re-center X again after this function call.
'''
def corrupt(feat_dim, n_dir, cor_portion, opt):

    prev_dir_l = []
    #noise_norm = opt.norm_scale*np.sqrt(feat_dim)    
    #noise_m = torch.from_numpy(ortho_group.rvs(dim=feat_dim).astype(np.float32)).to(device)
    #chunk_sz = n_cor // n_dir
    #cor_idx = torch.LongTensor(list(range(n_cor))).to(utils.device).unsqueeze(-1)
    
    noise_idx = 0
    #generate n_dir number of norms, sample in interval [kp, sqrt(d)]
    #for testing, to achieve high acc for tau0 & tau1: noise_norms = np.random.normal( np.sqrt(feat_dim), 1. , (int(np.ceil(n_c
    
    #min number of samples per noise dir
    n_noise_min = 520
    ##noise_vecs = torch.zeros(n_dir, feat_dim, device=X.device)
    
    ##base_n = int(float(n_cor)/sum([1.2**i for i in range(n_dir)]))
    end = 0
    noise_vecs_l = []
    chunk_sz = (feat_dim-1) // n_dir
    for i in range(n_dir):
        #start = end
        #end = min(n_cor, int(end + base_n*1.2**i))
        
        cur_n = int(n_noise_min * 1.1**i)
        cur_noise_vecs = 0.1 *torch.randn(cur_n, feat_dim).to(utils.device)
                
        cur_noise_vecs[:, i*chunk_sz] += np.sqrt(n_dir/np.clip(cor_portion, 0.01, None))
        #noise_vecs[start:end, noise_idx] += 1./np.clip(cor_portion, 0.01, None) 
        #cur_noise_vecs[:, i] += 1./np.clip(cor_portion, 0.01, None) #np.sqrt(feat_dim)/2
        
        cur_noise_vecs[cur_n//2:] *= (-1)
        ###corrupt1d(X, prev_dir_l, cor_idx[start:end], noise_vecs[start:end])        
        noise_vecs_l.append(cur_noise_vecs)
        
    #noise_vecs = 0.1 *torch.randn(n_cor, feat_dim, device=X.device)
    noise_vecs = torch.cat(noise_vecs_l, dim=0)
    cor_idx = torch.LongTensor(list(range(len(noise_vecs)))).to(utils.device)
    n = int(len(noise_vecs)/(cor_portion/(1-cor_portion)))
    X = generate_sample(n, feat_dim)
    X = torch.cat((noise_vecs, X), dim=0)
    if len(X) < feat_dim:
        print('Warning: number of samples smaller than feature dim!')
    '''
    idx = torch.zeros(n_dir, n_points, device=X.device)
    src = torch.ones(1, n_cor, device=X.device).expand(n_dir, -1)
    
    idx.scatter_add_(1, cor_idx, src)
    idx = idx.sum(0)
    cor_idx = torch.LongTensor(range(n_points))[idx.view(-1)>0].to(X.device)
    '''
    return X, cor_idx, noise_vecs

'''
Returns:
-noise: (1, feat_dim) noise vec
@deprecated
'''
def create_noise_dep(X, prev_dir_l):
    #with high prob, randomly generated vecs are orthogonal
    no_noise = True
    feat_dim = X.size(1)
    while no_noise:
        noise = torch.randn(1, feat_dim, device=X.device)
        too_close = False
        for d in prev_dir_l:
            if (d*noise).sum().item() > NOISE_INN_THRESH:
                too_close = True
                break
        if not too_close:
            no_noise = False
    return noise

'''
Modifies in-place.
Input:
-X: tensor
-n_dir: number of directions
-n_cor: number of points to be corrupted
-noise: 2D vec, (1, feat_dim)
'''
def corrupt1d(X, prev_dir_l, cor_idx, noise):
    n_points, feat_dim = X.size()

    prev_dir_l.append(noise)
    #add to a portion of samples        
    n_cor = cor_idx.size(0)

    ##print('indices corrupted {}'.format(cor_idx.view(-1)))
    #create index vec
    #e.g. [[1,1,...],[4,4,...]]
    idx = cor_idx.expand(n_cor, feat_dim) ##

    #add noise in opposing directions
    noise2dir = True    
    if noise2dir and n_cor > 1:
        noise_neg = -noise
        len0 = n_cor//2
        len1 = n_cor - len0        
        #X.scatter_(0, idx[:len0], noise.expand(len0, -1))
        #X.scatter_(0, idx[len0:], noise_neg.expand(len1, -1))
        X.scatter_(0, idx[:len0], noise[:len0] )
        X.scatter_(0, idx[len0:], noise_neg[len0:])
    else:
        X.scatter_(0, idx, noise.expand(n_cor, -1))
        X.scatter_(0, idx[len0:], noise)
        
    return cor_idx

'''
Create data samples
'''
def generate_sample(n, feat_dim):
    #create sample with mean 0 and variance 1
    X = torch.randn(n, feat_dim, device=device)
    #X = X/(X**2).sum(-1, keepdim=True)
    return X
    
'''
Compute top cov dir. To compute \tau_old
Returns:
-2D array, of shape (1, n_feat)
'''
def top_dir(X, opt, noise_vecs=None):
    X = X - X.mean(dim=0, keepdim=True)    
    X_cov = cov(X)
    if False:
        u, d, v_t = linalg.svd(X_cov.cpu().numpy())
        #pdb.set_trace()
        u = u[:opt.n_top_dir]        
    else:
        #convert to numpy tensor. 
        sv = decom.TruncatedSVD(opt.n_top_dir)
        sv.fit(X.cpu().numpy())
        u = sv.components_
    
    if noise_vecs is not None:
        
        print('inner of noise with top cov dirs')
        n_noise = noise_vecs.size(0)
        sv1 = decom.TruncatedSVD(n_noise)
        sv1.fit(X.cpu().numpy())
        u1 = torch.from_numpy(sv1.components_).to(device)
        print(utils.inner_mx(noise_vecs, u1)[:, :int(1.5*n_noise)])
    
    #U, D, V = svd(X, k=1)    
    return torch.from_numpy(u).to(device)
    
'''
Input:
-X: shape (n_sample, n_feat)
'''
def cov(X):    
    X = X - X.mean(dim=0, keepdim=True)    
    cov = torch.mm(X.t(), X) / X.size(0)
    return cov

'''
Compute accuracy.
Input:
-score: 1D tensor
-corrupt_idx: 1D tensor
Returns:
-percentage of highest-scoring points that are corrupt.
'''
def compute_acc(score, cor_idx):
    cor_idx = cor_idx.view(-1)
    n_idx = cor_idx.size(0)
    top_idx = torch.topk(score, k=n_idx)[1] #k
    #(1,k)
    top_idx = top_idx.unsqueeze(0).expand(n_idx, -1)
    cor_idx = cor_idx.unsqueeze(-1).expand(-1, n_idx)
        
    return float(top_idx.eq(cor_idx).sum()) / n_idx

'''
Compute accuracy with select index.
Input:
-score: 1D tensor
-corrupt_idx: 1D tensor
Returns:
-percentage of highest-scoring points that are corrupt.
'''
def compute_acc_with_idx(select_idx, cor_idx, X, n_removed):
    cor_idx = cor_idx.view(-1)
    n_idx = cor_idx.size(0)
    all_idx = torch.zeros(X.size(0), device=device) 
    ones = torch.ones(select_idx.size(0), device=device)
    all_idx.scatter_add_(dim=0, index=select_idx, src=ones)

    if device == 'cuda':
        try:
            range_idx = torch.cuda.LongTensor(range(X.size(0)))
        except RuntimeError:
            print('Run time error!')
            pdb.set_trace()
    else:
        range_idx = torch.LongTensor(range(X.size(0)))
    #(1,k)
    drop_idx = range_idx[all_idx == 0]    
    top_idx = drop_idx.unsqueeze(0).expand(n_idx, -1)
    
    '''
    top_idx = torch.topk(score, k=n_idx)[1] #k
    #(1,k)
    top_idx = top_idx.unsqueeze(0).expand(n_idx, -1)
    '''
    #(X.size(0), n_idx)
    cor_idx = cor_idx.unsqueeze(-1).expand(-1, n_removed)
    
    return float(top_idx.eq(cor_idx).sum()) / n_idx


def compute_tau1_fast(X, select_idx, opt, noise_vecs):
    
    X = torch.index_select(X, dim=0, index=select_idx)
    X_centered = X - X.mean(0, keepdim=True)

    if True:
        tau1 = utils.jl_chebyshev(X, opt.lamb)
    else:
        M = compute_m(X, opt.lamb, noise_vecs) 
        X_m = torch.mm(X_centered, M) #M should be symmetric, so not M.t()
        tau1 = (X_centered*X_m).sum(-1)   
    
    return tau1

'''
Input:
-X: centered
-select_idx: idx to keep for this iter, 1D tensor.
Output:
-X: updated X
-tau1
'''
def compute_tau1(X, select_idx, opt, noise_vecs):
    
    X = torch.index_select(X, dim=0, index=select_idx)
    #input should already be centered!
    X_centered = X - X.mean(0, keepdim=True)  
    M = compute_m(X, opt.lamb, noise_vecs) 
    X_m = torch.mm(X_centered, M) #M should be symmetric, so not M.t()
    tau1 = (X_centered*X_m).sum(-1)
        
    return tau1

'''
Input: already centered
'''
def compute_tau0(X, select_idx, opt, noise_vecs=None):
    X = torch.index_select(X, dim=0, index=select_idx)
    cov_dir = top_dir(X, opt, noise_vecs)
    #top dir can be > 1
    cov_dir = cov_dir.sum(dim=0, keepdim=True)
    tau0 = (torch.mm(cov_dir, X.t())**2).squeeze()    
    return tau0

'''
compute tau2, v^tM^{-1}v
'''
def compute_tau2(X, select_idx, opt, noise_vecs=None):
    X = torch.index_select(X, dim=0, index=select_idx)
    M = cov(X).cpu().numpy()
    M_inv = torch.from_numpy(linalg.pinv(M)).to(utils.device)
    scores = (torch.mm(X, M_inv)*X).sum(-1)
    #cov_dir = top_dir(X, opt, noise_vecs)    
    #top dir can be > 1
    #cov_dir = cov_dir.sum(dim=0, keepdim=True)
    #tau0 = (torch.mm(cov_dir, X.t())**2).squeeze()    
    return scores

'''
Input:
-X: input, already corrupted
-n: number of samples
'''
def train(X, noise_idx, outlier_method_l, opt):
    
    tau1, select_idx1, n_removed1, tau0, select_idx0, n_removed0 = compute_tau1_tau0(X, opt)
    
    all_idx = torch.zeros(X.size(0), device=device) 
    ones = torch.ones(noise_idx.size(0), device=device) 
    all_idx.scatter_add_(dim=0, index=noise_idx.squeeze(), src=ones)
        
    debug = False
    if debug:
        X_cov = cov(X)
        U, D, V_t = linalg.svd(X_cov.cpu().numpy())
        U1 = torch.from_numpy(U[0]).to(utils.device)
        '''        
        all_idx = torch.zeros(X.size(0), device=device)  #torch.cuda.LongTensor(range(X.size(0) )) #  dtype=torch.int64,
        ones = torch.ones(cor_idx.size(0), device=device) #dtype=torch.int64,        
        all_idx.scatter_add_(dim=0, index=cor_idx.squeeze(), src=ones)
        '''
        good_vecs = X[all_idx==0]
        w1_norm = (good_vecs**2).sum(-1).mean()
        good_proj = (good_vecs*U1).sum(-1)
        utils.hist(good_proj, 'inliers_syn', high=500)

        cor_vecs = torch.index_select(X, dim=0, index=cor_idx.squeeze())
        w2_norm = (cor_vecs**2).sum(-1).mean()
        cor_proj = (cor_vecs*U1).sum(-1)
        utils.hist(cor_proj, 'outliers_syn', high=500)
        pdb.set_trace()

    #scores of good and bad points
    good_scores1 = tau1[all_idx==0]
    bad_scores1 = tau1[all_idx==1]
    good_scores0 = tau0[all_idx==0]
    bad_scores0 = tau0[all_idx==1]
    
    auc1 = utils.auc(good_scores1, bad_scores1)
    auc0 = utils.auc(good_scores0, bad_scores0)
    print('auc0 {} auc1 {}'.format(auc0, auc1))
    scores_l = [auc1, auc0]
    
    #default is tau0
    for method in outlier_method_l:
        if method == 'iso forest':
            tau = baselines.isolation_forest(X)
        elif method == 'lof':
            tau = baselines.knn_dist_lof(X)
        elif method == 'ell env':
            tau = baselines.ellenv(X)
        elif method == 'dbscan':
            tau = baselines.dbscan(X)
        elif method == 'l2':
            tau = baselines.l2(X)
        elif method == 'knn':
            tau = baselines.knn_dist(X)
        else:
            raise Exception('method {} not supported'.format(method))
        
        good_tau = tau[all_idx==0]
        bad_tau = tau[all_idx==1]    
        auc = utils.auc(good_tau, bad_tau)        
        scores_l.append(auc)
        if opt.visualize_scores:
            pdb.set_trace()
            utils.inlier_outlier_hist(good_tau, bad_tau, method+'syn')
        
    #acc1 = compute_acc_with_idx(select_idx1, cor_idx, X, n_removed1)
    #acc0 = compute_acc_with_idx(select_idx0, cor_idx, X, n_removed0)
        
    return scores_l

'''
Computes tau1 and tau0.
Note: after calling this for multiple iterations, use select_idx rather than the scores tau 
for determining which have been selected as outliers. Since tau's are scores for remaining points after outliers.
Returns:
-tau1 and tau0, select indices for each, and n_removed for each
'''
def compute_tau1_tau0(X, opt):
    use_dom_eval = True
    if use_dom_eval:
        #dynamically set lamb now
        #find dominant eval.
        dom_eval, _ = utils.dominant_eval_cov(X)
        opt.lamb = 1./dom_eval * opt.lamb_multiplier        
        lamb = opt.lamb        

    #noise_vecs can be used for visualization.
    no_evec = True
    if no_evec:
        noise_vecs = None
        
    def get_select_idx(tau_method):
        if device == 'cuda':
            pdb.set_trace()
            select_idx = torch.cuda.LongTensor(list(range(X.size(0))))
        else:
            select_idx = torch.LongTensor(list(range(X.size(0))))
        n_removed = 0
        for _ in range(opt.n_iter):
            tau1 = tau_method(X, select_idx, opt, noise_vecs)
            #select idx to keep
            cur_select_idx = torch.topk(tau1, k=int(tau1.size(0)*(1-opt.remove_p)), largest=False)[1]
            #note these points are indices of current iteration            
            n_removed += (select_idx.size(0) - cur_select_idx.size(0))
            select_idx = torch.index_select(select_idx, index=cur_select_idx, dim=0)            
        return select_idx, n_removed, tau1

    if opt.fast_jl:
        select_idx1, n_removed1, tau1 = get_select_idx(compute_tau1_fast)
    else:
        select_idx1, n_removed1, tau1 = get_select_idx(compute_tau1)
        
    #acc1 = compute_acc_with_idx(select_idx, cor_idx, X, n_removed)    
    if DEBUG:
        print('new acc1 {}'.format(acc1))
        M = compute_m(X, opt.lamb, noise_vecs)
        X_centered = X - X.mean(0,keepdim=True)
        X_m = torch.mm(X_centered, M) #M should be symmetric, so not M.t()
        tau1 = (X_centered*X_m).sum(-1)
        print('old acc1 {}'.format(compute_acc(tau1, cor_idx)))
        pdb.set_trace()
    
    '''
    if device == 'cuda':
        select_idx = torch.cuda.LongTensor(range(X.size(0)))
    else:
        select_idx = torch.LongTensor(range(X.size(0)))
    for _ in range(opt.n_iter):
        tau0 = compute_tau0(X, select_idx, opt)
        cur_select_idx = torch.topk(tau0, k=tau1.size(0)*(1-opt.remove_p), largest=False)[1]
        select_idx = torch.index_select(select_idx, index=cur_select_idx, dim=0)
    '''
    select_idx0, n_removed0, tau0 = get_select_idx(compute_tau0)    
    
    return tau1, select_idx1, n_removed1, tau0, select_idx0, n_removed0

'''
Generate random data, corrupt, score, and test accuracies.
With respec to number of directions noise is added.
'''
def generate_and_score(opt, dataset_name='syn'):
    
    if opt.high_dim:
        opt.n = 2**15 
        opt.feat_dim = 8192
    else:
        opt.n = 10000 
        opt.feat_dim = 128 
    
    n = opt.n
    feat_dim = opt.feat_dim
    n_repeat = 2
    opt.p = 0.2 #default total portion corrupted
    #number of top dirs for calculating tau0
    opt.n_top_dir = 1
    opt.dataset_name = dataset_name
    data_l = []
    n_dir_l = list(range(1, 16, 3))
    n_dir_l = [2]
    if dataset_name == 'syn':
        for n_dir in n_dir_l:
            cur_data_l = []
            for _ in range(n_repeat):
                X, cor_idx, noise_vecs = corrupt(feat_dim, n_dir, opt.p, opt)
                X = X.to(device='cpu')
                n = len(X)
                
                X = X - X.mean(0)
                if opt.fast_jl:
                    X = utils.pad_to_2power(X)                
                cur_data_l.append([X, cor_idx])        

                '''
                X = generate_sample(n, feat_dim)
                cor_idx, noise_vecs = corrupt(X, n_dir, cor_portion, opt)
                X = X - X.mean(0)
                if opt.fast_jl:
                    X = utils.pad_to_2power(X)
                data_l.append([X, cor_idx])
                '''
            data_l.append(cur_data_l)
        
    elif dataset_name == 'glove':
        _, X = data.process_glove_data()
        X = X[:50000]
    else:
        X = torch.load('data/val_embs0.pt').to(utils.device)
        X = X - X.mean(0)
        n, feat_dim = X.size(0), X.size(1)
        
    print('samples size {} {} padded: {}'.format(n, feat_dim, X.size(1)))
        
    plot_lambda = True

    #Note now lamb set based on proportion corrupted later on. Setting here is void.
    opt.lamb = None 
    lamb = opt.lamb
    #which baseline to use as tau0, can be 'isolation_forest', 'l2', 'tau0' etc
    opt.baseline = 'tau0'
    
    opt.n_iter = 1
    #amount to remove wrt cur_p
    opt.remove_factor = 1./opt.n_iter #0.5
        
    #scalar to multiply norm of noise vectors with. This is deprecated
    opt.norm_scale = 1.3
    #amount to divide noise norm by
    opt.noise_norm_div = 8
    acc_l = []
    #numpy array used for plotting.
    k_l = []
    p_l = []
    tau_l = []
    res_l = []
    
    outlier_methods_l = ['l2', 'lof', 'isolation_forest', 'knn', 'dbscan']
    outlier_methods_l = ['l2', 'iso forest', 'ell env', 'lof', 'knn']
    outlier_methods_l = ['knn'] #march 
    #+3 for tau1 tau0 and lamb
    scores_ar = np.zeros((len(n_dir_l), len(outlier_methods_l)+3))
    std_ar = np.zeros((len(n_dir_l), len(outlier_methods_l)+3))
    opt.lamb_multiplier = 4
    
    for j, n_noise_dir in enumerate(n_dir_l):

        opt.n_dir = n_noise_dir
        #percentage to remove   
        opt.remove_p = opt.p*opt.remove_factor
        
        cur_res_l = [n, feat_dim, n_noise_dir, opt.p, opt.lamb_multiplier, opt.norm_scale]
        acc_mx = torch.zeros(n_repeat, 2)
        cur_scores_ar = np.zeros((n_repeat, len(outlier_methods_l)+2))
        cur_data_l = data_l[j]
        for i in range(n_repeat):            
            X, noise_idx = cur_data_l[i]
            X = X.to(device=utils.device)
            cur_scores_ar[i] = train(X, noise_idx, outlier_methods_l, opt)
            acc_mx[i, 0] = cur_scores_ar[i, 1] #acc0
            acc_mx[i, 1] = cur_scores_ar[i, 0] #acc1
            n_data = len(X)
            del(X)
            
        print(n_data)
        take_diff = False
        if take_diff:
            cur_scores_ar[:, 1] = cur_scores_ar[:, 0] - cur_scores_ar[:, 1]
            cur_scores_ar[:, 2] = cur_scores_ar[:, 0] - cur_scores_ar[:, 2]
        scores_ar[j, 1:] = np.mean(cur_scores_ar, axis=0)
        
        if opt.use_std:
            std_ar[j, 1:] = np.std(cur_scores_ar, axis=0)
        else:
            #.95 confidence intervals
            se = np.clip(st.sem(cur_scores_ar, axis=0), 1e-3, None)        
            low, high = st.t.interval(0.95, cur_scores_ar.shape[0]-1, loc=scores_ar[j, 1:], scale=se)
            std_ar[j, 1:] = (high - low)/2.
            
        scores_ar[j, 0] = n_noise_dir
        acc_mean = acc_mx.mean(dim=0)
        acc0, acc1 = acc_mean[0].item(), acc_mean[1].item()
        print('n_noise_dir {} lamb {} acc0 {} acc1 {}'.format(n_noise_dir, opt.lamb_multiplier, acc0, acc1))
        cur_res_l.extend([acc0, acc1])
        p_l.extend([lamb, lamb]) #[np.around(cur_p, decimals=2)]*2)
        k_l.extend([n_noise_dir, n_noise_dir])
        acc_l.extend([acc0, acc1])        
        tau_l.extend([0, 1])
        res_l.append(cur_res_l)
    
    print('About to plot!')
    pdb.set_trace()
    if plot_lambda:
        legends = ['k', 'acc', 'tau', 'lambda']
    else:
        legends = ['k', 'acc', 'tau', 'p']
    utils.plot_acc(k_l, acc_l, tau_l, p_l, legends, opt)
    scores_ar = scores_ar.transpose()
    std_ar = std_ar.transpose()
    utils.plot_scatter_flex(scores_ar, ['tau1', 'tau0'] + outlier_methods_l, opt, std_ar=std_ar)
    m = {'opt':opt, 'scores_ar':scores_ar, 'conf_ar':std_ar}
    f_name = 'dirs_data_fast.npy' if opt.fast_jl else 'dirs_data.npy' 
    with open(osp.join('results', opt.dir, f_name), 'wb') as f:
        torch.save(m, f)
        print('saved under {}'.format(f))

    write_results = False
    if write_results:
        #k_l, acc_l, tau_l, p_l,
        res_path = osp.join(utils.res_dir, 'acc_res.txt')
        res_l.insert(0, repr(opt))
        res_l.insert(0, str(legends))
        utils.write_lines(res_l, res_path, 'a')

'''
Vary synthetic data wrt lambda.
'''
def generate_and_score_lamb(opt, dataset_name='syn'):

    n_dir_l = [3, 6, 10]
    n_dir_l = [3, 6, 10]
    
    legend_l = []
    scores_l = []
    conf_l = []
    if opt.compute_scores_diff:
        for n_dir in n_dir_l:
            legend_l.append(str(n_dir))
            opt.n_dir = n_dir
            mean1, conf1 = generate_and_score_lamb2(opt, dataset_name)
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

        tau1_ar = np.concatenate((mean1[0].reshape(1, -1), scores_ar[:,1,:].reshape(len(n_dir_l), n_lamb)), axis=0)
        tau1_conf_ar = np.concatenate((mean1[0].reshape(1, -1), conf_ar[:,1,:].reshape(len(n_dir_l), n_lamb)), axis=0)
        #scores_ar = np.concatenate((mean1[:, 0].reshape(1, -1), scores_ar[:,:,3]), axis=0)
        #scores_ar = np.stack([mean1[:, 0]]+conf_l, axis=0)
        #np.concatenate((mean1[:, 0].reshape(1,-1), np.stack(scores_l, axis=0)), axis=0)
        pdb.set_trace()
        utils.plot_scatter_flex(tau0_ar, legend_l, opt, std_ar=tau0_conf_ar, name='tau0')
        utils.plot_scatter_flex(l2_ar, legend_l, opt, std_ar=l2_conf_ar, name='l2')

        m = {'opt':opt, 'scores_ar':scores_ar, 'conf_ar':conf_ar}
        with open(osp.join('results', opt.dir, 'lamb_data.npy'), 'wb') as f:
            torch.save(m, f)
            print('saved under {}'.format(f))
    else:
        for n_dir in n_dir_l:
            legend_l.append(str(n_dir))
            opt.n_dir = n_dir
            mean1, conf1 = generate_and_score_lamb2(opt, dataset_name)
            scores_l.append(mean1[:, 1])
            conf_l.append(conf1[:, 1])

        scores_ar = np.concatenate((mean1[:, 0].reshape(1,-1), np.stack(scores_l, axis=0)), axis=0)
        conf_ar = np.concatenate((mean1[:, 0].reshape(1,-1), np.stack(conf_l, axis=0)), axis=0)
        pdb.set_trace()
        utils.plot_scatter_flex(scores_ar, legend_l, opt, std_ar=conf_ar)
    
'''
Vary wrt lambda.
'''
def generate_and_score_lamb2(opt, dataset_name='syn'):

    if opt.high_dim:
        opt.n = 2**15 
        opt.feat_dim = 8192 
    else:
        opt.n = 8000 #10000 #50
        opt.feat_dim = 128 #1024 #1400 #1000
        
    n = opt.n
    feat_dim = opt.feat_dim
    #number of top dirs for calculating tau0
    opt.n_top_dir = 1
    opt.dataset_name = dataset_name
    n_repeat = 2
    cor_portion = .2

    lamb_l = list(range(5, 31, 5))
    lamb_l = list(range(0, 16, 5))
    #lamb_l = [20]
    
    data_l = []
    if dataset_name == 'syn':
        for _ in range(n_repeat):            
            #X = generate_sample(n, feat_dim)
            #cor_idx, noise_vecs = corrupt(X, opt.n_dir, cor_portion, opt)
            
            X, cor_idx, noise_vecs = corrupt(feat_dim, opt.n_dir, cor_portion, opt)
            n = len(X)
            X = X - X.mean(0)
            if opt.fast_jl:
                X = utils.pad_to_2power(X)                
            data_l.append([X, cor_idx])        
    elif dataset_name == 'glove':
        _, X = data.process_glove_data()
        X = X[:50000]
    else:
        X = torch.load('data/val_embs0.pt').to(utils.device)
        X = X - X.mean(0)
        n, feat_dim = X.size(0), X.size(1)
    
    print('samples size {} {} padded: {}'.format(n, feat_dim, X.size(1)))
        
    #which baseline to use as tau0, can be 'isolation_forest', la, tau0, etc
    opt.baseline = 'tau0'
    print('baseline method: {}'.format(opt.baseline))
    
    opt.n_iter = 1
    #amount to remove wrt cur_p
    opt.remove_factor = 1./opt.n_iter
    opt.p = 0.2 #default total portion corrupted
    
    #scalar to multiply norm of noise vectors with. This is deprecated
    opt.norm_scale = 1.3
    #amount to divide noise norm by, deprecated
    opt.noise_norm_div = 8
    
    #opt.n_dir = N_DIR
    #n_dir = opt.n_dir
    acc_l = []
    #numpy array used for plotting.
    k_l = []
    p_l = []
    tau_l = []
    res_l = []
    
    outlier_methods_l = ['l2', 'iso forest', 'ell env', 'lof', 'knn']
    #only compare with l2 (and tau0) when studying effects of lambda
    outlier_methods_l = ['l2']

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
            #includes tau1 and outlier_method_l baseline scores
            cur_scores_ar[i] = train(X, noise_idx, outlier_methods_l, opt)
            acc_mx[i, 0] = cur_scores_ar[i, 1]
            acc_mx[i, 1] = cur_scores_ar[i, 0]
        
        if opt.compute_scores_diff:
            #whether to use tau1 raw scores or compute tau1 - tau0 and tau1 - l2 
            cur_scores_ar[:, 1] = cur_scores_ar[:, 0] - cur_scores_ar[:, 1]
            cur_scores_ar[:, 2] = cur_scores_ar[:, 0] - cur_scores_ar[:, 2]
        
        scores_ar[j, 1:] = np.mean(cur_scores_ar, axis=0)
        #whether to use std vs confidence interval        
        if opt.use_std:
            std_ar[j, 1:] = np.std(cur_scores_ar, axis=0)
        else:
            se = np.clip(st.sem(cur_scores_ar, axis=0), 1e-4, None)        
            low, high = st.t.interval(0.95, cur_scores_ar.shape[0]-1, loc=scores_ar[j, 1:], scale=se)        
            std_ar[j, 1:] = (high - low)/2.        
        
        scores_ar[j, 0] = lamb        
        acc_mean = acc_mx.mean(dim=0)
        acc0, acc1 = acc_mean[0].item(), acc_mean[1].item()
        print('n_noise_dir {} lamb {} acc0 {} acc1 {}'.format(opt.n_dir, lamb, acc0, acc1))
        
    scores_ar = scores_ar.transpose()
    std_ar = std_ar.transpose()
    print(std_ar)
    plot = False
    if plot:
        print('About to plot!')
        pdb.set_trace()
        legends = ['lamb', 'acc', 'tau']
        #plot both tau1 vs tau0, and tau1 against all baselines.
        utils.plot_acc_syn_lamb(p_l, acc_l, tau_l, legends, opt)
        
        utils.plot_scatter_flex(scores_ar, ['tau1', 'tau0'] + outlier_methods_l, opt, std_ar=std_ar)
    
    write_res = False
    if write_res:
        res_path = osp.join(utils.res_dir, 'acc_res.txt')
        res_l.insert(0, repr(opt))
        res_l.insert(0, str(legends))
        utils.write_lines(res_l, res_path, 'a')
    return scores_ar, std_ar

'''
Plot baseline methods for input data, along with input tau1 and tau0 scores.
Input:
-scores_l: list of tensors of scores, e.g. rocauc or accuracy
-legend_l: list of legends.
'''
def plot_base_lines(data_l, legend_l, opt, std_l=None):
    
    for i, scores in enumerate(data_l):        
        data_l[i] = scores.cpu().numpy()
                
    utils.plot_scatter_flex(data_l, legend_l, opt, std_l)
    
    
'''
Use genetics data
'''
def test_genetics_data():
    X = data.load_genetics_data()    
    #X = torch.from_numpy(X).to(dtype=torch.float32, device=utils.device)
    X = utils.pad_to_2power(X)
    
    opt = utils.parse_args()    
    opt.n, opt.feat_dim = X.size(0), X.size(1)
    #percentage of points to remove
    opt.remove_p = 0.1
    opt.baseline = 'tau0'
    
    #number of top dirs for calculating tau0
    opt.n_top_dir = 1
    
    print('samples size {} {}'.format(opt.n, opt.feat_dim))
    opt.n_iter = 1    
    tau1, select_idx1, n_removed1, tau0, select_idx0, n_removed0 = compute_tau1_tau0(X, opt)
    pdb.set_trace()


def test_glove_data(opt):

    text_name = 'sherlock'
    
    with open('data/{}.txt'.format(text_name), 'r') as file:
    #with open('data/tolstoy.txt', 'r') as file:
        lines = file.readlines()
        
    lines_len = len(lines)
    n_lines = 800
    opt.lamb_multiplier = 20 #20
    
    #tau0_percent_l, tau1_percent_l = [], []
    tau0_percent, tau1_percent, auc_prob0, auc_prob1 = 0, 0, 0, 0
    counter = 0
    outlier_method_l = []
    outlier_methods_l = ['l2', 'iso forest', 'ell env', 'lof', 'knn']
    for i in range(0, lines_len, n_lines):
        cur_lines = lines[i : min(lines_len, i+n_lines)]
        scores_l = test_glove_data2(cur_lines, None, outlier_method_l, opt)    
        cur_tau0_percent, cur_tau1_percent, cur_auc_prob0, cur_auc_prob1 = scores_l[:4]
        #tau0_percent_l.append(tau0_percent)
        #tau1_percent_l.append(tau1_percent)
        tau0_percent += cur_tau0_percent
        tau1_percent += cur_tau1_percent
        auc_prob0 += cur_auc_prob0
        auc_prob1 += cur_auc_prob1
        
        if counter == 30:
            break
        counter += 1
        
    n_times = counter+1 #((lines_len-1)//n_lines + 1)
    
    tau0_percent /= n_times
    tau1_percent /= n_times
    
    print('tau0_percent {} tau1_percent {}'.format(tau0_percent, tau1_percent))
    print('tau0_ prob {} tau1_prob {}'.format(auc_prob0/n_times, auc_prob1/n_times))

'''
Test on glove data with various lambdas.
'''
def test_glove_data_lamb_dep():

    opt = utils.parse_args()
    #name can be 'tolstoy', 'sherlock'
    opt.text_name = 'sherlock' #'tolstoy'
    #with open('data/sherlock.txt', 'r') as file:
    with open('data/{}.txt'.format(opt.text_name), 'r') as file:
        lines = file.readlines()
        
    lines_len = len(lines)
    n_lines = 800
    
    tau0_percent, tau1_percent, auc_prob0, auc_prob1 = 0, 0, 0, 0
    
    tau0_percent_l, tau1_percent_l, prob1_l = [], [], []
    mult = list(range(3, 15, 3))
    outlier_method_l = []
    for i in mult: #(0, 30, 6) too coarse already
        #can do average of several runs
        opt.lamb_multiplier = i
        cur_tau0_percent, cur_tau1_percent, cur_auc_prob0, cur_auc_prob1 = 0, 0, 0, 0
        counter = 0
        for i in range(0, lines_len, n_lines):
            cur_lines = lines[i : min(lines_len, i+n_lines)]
    
            cur_tau0_percent0, cur_tau1_percent0, cur_auc_prob00, cur_auc_prob10 = test_glove_data2(cur_lines, opt)
            cur_tau0_percent += cur_tau0_percent0
            cur_tau1_percent += cur_tau1_percent0
            cur_auc_prob0 += cur_auc_prob00
            cur_auc_prob1 += cur_auc_prob10
            counter += 1
            if counter == 5:
                break
    
        tau0_percent = cur_tau0_percent/counter
        tau1_percent = cur_tau1_percent/counter
        auc_prob0 = cur_auc_prob0/counter
        auc_prob1 = cur_auc_prob1/counter
        tau1_percent_l.append(tau1_percent)
        tau0_percent_l.append(tau0_percent)
        prob1_l.append(auc_prob1)
        
    '''
    n_times = counter+1 #((lines_len-1)//n_lines + 1)    
    tau0_percent /= n_times
    tau1_percent /= n_times
    '''
    print('tau0_percent {} tau1_percent {} prob {}'.format(tau0_percent_l, tau1_percent_l, prob1_l))
    print('About to plot!')
    pdb.set_trace()
    utils.plot_scatter(mult, tau1_percent_l, ['lambda_multiplier', 'recall'], opt)
    utils.plot_scatter(mult, prob1_l, ['lambda_multiplier', 'rocauc'], opt)
    #print('tau0_ prob {} tau1_prob {}'.format(auc_prob0/n_times, auc_prob1/n_times))

def test_glove_data_lamb(opt, noise_percent=0.2):

    #don't use all 14
    n_dir_l = [3, 6, 10]
    legend_l = []
    mean_l = []
    conf_l = []
    scores_l = []
    conf_l = []
    if opt.compute_scores_diff:
        for n_dir in n_dir_l:
            legend_l.append(str(n_dir))
            opt.n_dir = n_dir
            mean1, conf1 = test_glove_data_lamb2(opt, noise_percent)
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
        m = {'opt':opt, 'scores_ar':scores_ar, 'conf_ar':conf_ar}
        with open(osp.join('results', opt.dir, 'lamb_data.npy'), 'wb') as f:
            torch.save(m, f)
            print('saved under {}'.format(f))

    else:
        raise Exception('Should compute diff instead of raw scores')
        for n_dir in n_dir_l:
            legend_l.append(str(n_dir))
            opt.n_dir = n_dir
            lamb_l, mean1, conf1 = test_glove_data_lamb2(opt, noise_percent)
            mean_l.append(mean1)
            conf_l.append(conf1)

        mean_ar = np.stack([lamb_l] + mean_l, axis=0)
        conf_ar = np.stack([lamb_l] + conf_l, axis=0)

        print('About to overwrite plot!')
        pdb.set_trace()
        utils.plot_scatter_flex(mean_ar, legend_l, opt, std_ar=conf_ar)

    
def test_glove_data_lamb2(opt, noise_percent=0.2):

    opt.text_name = 'sherlock' #'sherlock'
    with open('data/{}.txt'.format(opt.text_name), 'r') as file:
        lines = file.readlines()
    text_str = ' '.join(lines)
    text_str_len = len(text_str)
    
    counter = 0
    
    total_noise_len = 0
    max_noise_len = 4300
    all_noise_lines = []
    n_repeat = 20
    
    for j in range(0, opt.n_dir):
        cur_lines = utils.read_lines('data/wiki_noise{}.txt'.format(j))            
        cur_noise_str = ' '.join(cur_lines)        
        cur_noise_str = cur_noise_str[:int(max_noise_len/1.2**j)]            
        all_noise_lines.append(cur_noise_str)
        total_noise_len += len(cur_noise_str)
    all_noise_lines = [' '.join(all_noise_lines)]
    
    '''
    noise_lines0 = utils.read_lines('data/sherlock_noise3.txt')
    noise_lines0.extend(utils.read_lines('data/sherlock_noise2.txt'))
    noise_lines0 = utils.read_lines('data/sherlock_noise5.txt')
    noise_str = ''.join(noise_lines0)
    total_noise_len = len(noise_str)
    all_noise_lines = [noise_str]
    '''
    
    tau0_percent_l, tau1_percent_l, prob0_l, prob1_l = [], [], [], []
    tau0_percent_std_l, tau1_percent_std_l, prob0_std_l, prob1_std_l = [], [], [], []
    #fix lambda_multiplier, vary the number of noise directions
    #opt.lamb_multiplier = 5 #10    
    mult_l = list(range(0, 17, 4))
    mult_l = list(range(0, 21, 4))
    outlier_method_l = ['knn']
    outlier_methods_l = ['l2', 'iso forest', 'ell env', 'lof', 'knn']
    #only run against l2 when studying dependence on lambda
    outlier_method_l = ['l2']
    scores_ar = np.zeros((len(mult_l), len(outlier_method_l)+2))
    std_ar = np.zeros((len(mult_l), len(outlier_method_l)+2))
    for i, lamb in enumerate(mult_l): #(0, 30, 6) too coarse already
        print('** CURRENT lamb {}'.format(lamb))
        opt.lamb_multiplier = lamb        
        cur_tau0_percent_l, cur_tau1_percent_l, cur_auc_prob0_l, cur_auc_prob1_l = [], [], [], []
        counter = 0
        
        #iterate over both content and noise.
        content_len = int(total_noise_len / (noise_percent/(1-noise_percent)))
        
        cur_scores_ar = np.zeros((n_repeat, len(outlier_method_l)+2))
        for k in range(n_repeat):
        #for i in range(0, text_str_len, content_len):
            start = random.randint(0, text_str_len-content_len)
            #iterate over different noises
            content_lines = [text_str[start:start+content_len]]
                        
            scores_l = test_glove_data2(content_lines, all_noise_lines, outlier_method_l, opt)
            cur_scores_ar[k] = scores_l[2:]
            cur_tau0_percent0, cur_tau1_percent0, cur_auc_prob00, cur_auc_prob10 = scores_l[:4]
            cur_tau0_percent_l.append(cur_tau0_percent0)
            cur_tau1_percent_l.append(cur_tau1_percent0)
            cur_auc_prob0_l.append(cur_auc_prob00)
            cur_auc_prob1_l.append(cur_auc_prob10)
        counter += 1
        if counter == 4:
            break
        
        tau0_percent, tau0_percent_std = np.mean(cur_tau0_percent_l), np.std(cur_tau0_percent_l)        
        tau1_percent, tau1_percent_std = np.mean(cur_tau1_percent_l), np.std(cur_tau1_percent_l)
        #cur_auc_prob0_l

        #std_ar[j, 1:] = st.t.interval(0.95, cur_scores_ar.shape[0]-1, loc=scores_ar[j, 1:], scale=se)        
        auc_prob0 = np.mean(cur_auc_prob0_l)
        conf_int0 = st.t.interval(0.95, len(cur_auc_prob0_l)-1, loc=auc_prob0, scale=st.sem(cur_auc_prob0_l))

        temp = np.array(cur_scores_ar[:, 0])        
        cur_scores_ar[:, 0] = cur_scores_ar[:, 1]
        cur_scores_ar[:, 1] = temp
        if opt.compute_scores_diff:
            #tau1 - tau0
            cur_scores_ar[:, 1] = cur_scores_ar[:, 0] - cur_scores_ar[:, 1]
            cur_scores_ar[:, 2] = cur_scores_ar[:, 0] - cur_scores_ar[:, 2]
        scores_ar[i] = cur_scores_ar.mean(axis=0)
        auc_prob1 = np.mean(cur_auc_prob1_l)        
        if opt.use_std:
            std_ar[i] = cur_scores_ar.std(axis=0)
            conf_int1 = np.std(cur_auc_prob1_l)
        else:            
            low, high = st.t.interval(0.95, n_repeat-1, loc=auc_prob1, scale=st.sem(cur_auc_prob1_l))
            conf_int1 = (high - low)/2.
            se = st.sem(cur_scores_ar, axis=0)
            low, high = st.t.interval(0.95, n_repeat-1, loc=scores_ar[i], scale=se)
            std_ar[i] = (high - low)/2.
        
        #auc_prob1, auc_prob1_std = np.mean(cur_auc_prob1_l), np.std(cur_auc_prob1_l)
                
        tau1_percent_l.append(tau1_percent)
        tau0_percent_l.append(tau0_percent)
        prob0_l.append(auc_prob0)
        prob1_l.append(auc_prob1)
        tau1_percent_std_l.append(tau1_percent_std)
        tau0_percent_std_l.append(tau0_percent_std)
        prob0_std_l.append(conf_int0)
        prob1_std_l.append(conf_int1)    
        
    print('tau0_percent {} tau1_percent {} \n prob {} {}'.format(tau0_percent_l, tau1_percent_l, prob0_l, prob1_l))
    print('Standard deviations {} {}'.format(prob0_std_l, prob1_std_l))
    plot = False
    if plot:
        pdb.set_trace()
        utils.plot_scatter(mult_l, prob1_l, ['lamb', 'rocauc_tau1'], opt, std=prob1_std_l)
        utils.plot_scatter(mult_l, prob0_l, ['lamb', 'rocauc_tau0'], opt, std=prob0_std_l)

    mult_ar = np.array(mult_l).reshape(1, -1)
    scores_ar = np.concatenate((mult_ar, scores_ar.transpose()), axis=0)
    std_ar = np.concatenate((mult_ar, std_ar.transpose()), axis=0)
    #return np.array(mult_l), np.array(prob1_l), np.array(prob1_std_l)
    return scores_ar, std_ar
    
'''
Different outlier directions. Increasingly more outlier directions.
But cap the total noise percentage to threshold.
'''
def test_glove_data_dirs(opt, noise_percent=0.2):

    #opt = utils.parse_args()

    opt.text_name = 'sherlock' #'sherlock'
    with open('data/{}.txt'.format(opt.text_name), 'r') as file:
        lines = file.readlines()
    text_str = ' '.join(lines)
    text_str_len = len(text_str)
    lines_len = len(lines)
    #initial estimate of content lines in order to determine noise.
    #n_lines = 1000 if noise_percent<0.11 else 1600
    
    tau0_percent, tau1_percent, auc_prob0, auc_prob1 = 0, 0, 0, 0
    counter = 0
    #cur_lines = lines[0 : min(lines_len, n_lines)]
    
    tau0_percent_l, tau1_percent_l, prob0_l, prob1_l = [], [], [], []
    tau0_percent_std_l, tau1_percent_std_l, prob0_std_l, prob1_std_l = [], [], [], []
    #fix lambda_multiplier, vary the number of noise directions
    opt.lamb_multiplier = 4 #10
    max_noise_len = 4000
    print('lamb_multiplier {}'.format(opt.lamb_multiplier))
    mult = list(range(1, 9))
    #outlier_method_l = ['l2', ] #may
    
    outlier_method_l = ['l2', 'iso forest', 'ell env', 'lof', 'knn']
    scores_ar = np.zeros((len(mult), len(outlier_method_l)+3))
    std_ar = np.zeros((len(mult), len(outlier_method_l)+3))    
    
    for k, n_dir in enumerate(mult): #(0, 30, 6) too coarse already
        print('CURRENT n_dir {}'.format(n_dir))
        all_noise_lines = []
        total_noise_len = 0
        
        for j in range(0, n_dir+1):
            cur_lines = utils.read_lines('data/wiki_noise{}.txt'.format(j))            
            cur_noise_str = ' '.join(cur_lines)
            cur_noise_str = cur_noise_str[:int(max_noise_len/1.2**j)]            
            all_noise_lines.append(cur_noise_str)
            total_noise_len += len(cur_noise_str)
            
        '''    
        #trim noise lines to be within noise_percent
        augment_ratio = noise_percent / (n_noise_lines/float(n_lines))
        
        new_noise_lines = []
        for cur_noise_lines in all_noise_lines:
            #if augment_ratio>1 all lines are appended.
            upto = int(augment_ratio*len(cur_noise_lines))
            new_noise_lines.extend(cur_noise_lines[:upto])
            
        all_noise_lines = new_noise_lines
        '''        
        cur_tau0_percent_l, cur_tau1_percent_l, cur_auc_prob0_l, cur_auc_prob1_l = [], [], [], []
                
        content_len = int(total_noise_len / (noise_percent/(1-noise_percent)))
        counter = 0
        
        cur_scores_ar = np.zeros((int(text_str_len/content_len), len(outlier_method_l)+2))        
        for i, start in enumerate(range(0, text_str_len, content_len)):
            if start+content_len > text_str_len:
                break
            #content_lines = lines[i : min(lines_len, i+n_lines)]
            content_lines = [text_str[start : start+content_len]]
            scores_l = test_glove_data2(content_lines, all_noise_lines, outlier_method_l, opt)
            cur_tau0_percent0, cur_tau1_percent0, cur_auc_prob00, cur_auc_prob10 = scores_l[:4]
            cur_scores_ar[i, 0] = cur_auc_prob10
            cur_scores_ar[i, 1] = cur_auc_prob00
            cur_scores_ar[i, 2:] = scores_l[4:]
            cur_tau0_percent_l.append(cur_tau0_percent0)
            cur_tau1_percent_l.append(cur_tau1_percent0)
            cur_auc_prob0_l.append(cur_auc_prob00)
            cur_auc_prob1_l.append(cur_auc_prob10)
            counter += 1
            
        cur_scores_ar = cur_scores_ar[:counter]
        scores_ar[k, 1:] = np.mean(cur_scores_ar, axis=0)
        std_ar[k, 1:] = np.std(cur_scores_ar, axis=0)
        scores_ar[k, 0] = n_dir
        std_ar[k, 0] = n_dir
        
        tau0_percent, tau0_percent_std = np.mean(cur_tau0_percent_l), np.std(cur_tau0_percent_l)        
        tau1_percent, tau1_percent_std = np.mean(cur_tau1_percent_l), np.std(cur_tau1_percent_l)        
        auc_prob0, auc_prob0_std = np.mean(cur_auc_prob0_l), np.std(cur_auc_prob0_l)
        auc_prob1, auc_prob1_std = np.mean(cur_auc_prob1_l), np.std(cur_auc_prob1_l)
        
        tau1_percent_l.append(tau1_percent)
        tau0_percent_l.append(tau0_percent)
        prob0_l.append(auc_prob0)
        prob1_l.append(auc_prob1)
        tau1_percent_std_l.append(tau1_percent_std)
        tau0_percent_std_l.append(tau0_percent_std)
        prob0_std_l.append(auc_prob0_std)
        prob1_std_l.append(auc_prob1_std)        
        
    print('tau0_percent {} tau1_percent {} prob {} {}'.format(tau0_percent_l, tau1_percent_l, prob0_l, prob1_l))
    pdb.set_trace()
    utils.plot_scatter(mult, prob1_l, ['n_dir', 'rocauc_tau1'], opt, std=prob1_std_l)
    utils.plot_scatter(mult, prob0_l, ['n_dir', 'rocauc_tau0'], opt, std=prob0_std_l)
    scores_ar = scores_ar.transpose()
    std_ar = std_ar.transpose()
    utils.plot_scatter_flex(scores_ar, ['tau1', 'tau0'] + outlier_method_l, opt, std_ar=std_ar)
    m = {'opt':opt, 'scores_ar':scores_ar, 'conf_ar':std_ar}
    with open(osp.join('results', opt.dir, 'dirs_data.npy'), 'wb') as f:
        torch.save(m, f)
        print('saved under {}'.format(f))

    #utils.plot_scatter(mult, tau1_percent_l, ['n_dir', 'recall_tau1'], opt, std=tau1_percent_std_l)    
    #utils.plot_scatter(mult, tau0_percent_l, ['n_dir', 'recall_tau0'], opt, std=tau0_percent_std_l)
    
    #print('tau0_ prob {} tau1_prob {}'.format(auc_prob0/n_times, auc_prob1/n_times))

'''
glove data generate scores.
'''
def test_glove_data2(content_lines, noise_lines, outlier_method_l, opt):
    
    ###words_ar, X = data.process_glove_data()
    content_path = 'data/sherlock.txt' if content_lines is None else None    
    noise_path = 'data/sherlock_noise3.txt' if noise_lines is None else None
    
    words_ar, X, noise_idx = words.doc_word_embed_content_noise(content_path, noise_path, 'data/sherlock_whiten.txt', content_lines, noise_lines, opt=opt)#.to(utils.device) #('data/sherlock_noise3.txt', 'data/test_noise.txt')#.to(utils.device)
    noise_idx = noise_idx.unsqueeze(-1)
    print('** {} number of outliers {}'.format(X.size(0), len(noise_idx)))
    #pdb.set_trace()
    X = X - X.mean(0)    
    X = utils.pad_to_2power(X)
    
    opt.n, opt.feat_dim = X.size(0), X.size(1)
    #percentage of points to remove.
    opt.remove_p = 0.2
    #number of top dirs for calculating tau0.
    opt.n_top_dir = 1
    opt.n_iter = 1 #13
    #use select_idx rather than the scores tau! Since tau's are scores for remaining points after outliers.
    tau1, select_idx1, n_removed1, tau0, select_idx0, n_removed0 = compute_tau1_tau0(X, opt)

    opt.baseline = 'tau0'
    scores_l = []
    
    all_idx = torch.zeros(X.size(0), device=device) 
    ones = torch.ones(noise_idx.size(0), device=device) 
    all_idx.scatter_add_(dim=0, index=noise_idx.squeeze(), src=ones)
    
    for method in outlier_method_l:
        if method == 'iso forest':
            tau = baselines.isolation_forest(X)
        elif method == 'lof':
            tau = baselines.knn_dist_lof(X)
        elif method == 'dbscan':
            tau = baselines.dbscan(X)
        elif method == 'ell env':
            tau = baselines.ellenv(X)
        elif method == 'l2':
            tau = baselines.l2(X)
        elif method == 'knn':
            tau = baselines.knn_dist(X)        
        elif method == 'tau2':
            select_idx2 = torch.LongTensor(list(range(len(X)))).to(utils.device)
            tau = compute_tau2(X, select_idx2, opt)
        else:
            raise Exception('method {} not supported'.format(method))
        if opt.visualize_scores:
            zeros = torch.zeros(len(tau), device=utils.device)
            zeros[noise_idx] = 1
            inliers_tau = tau[zeros==0] 
            outliers_tau = tau[zeros==1] 
            pdb.set_trace()
            utils.inlier_outlier_hist(inliers_tau, outliers_tau, 'text'+method, high=20)
        
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
            
    #complement of noise_idx            
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
    
    #selected_words1 = [words_ar[i] for i in outliers_idx1.cpu().numpy()]
    #selected_words0 = [words_ar[i] for i in outliers_idx0.cpu().numpy()]
    
    return [tau0_cor.item()/len(outliers_idx0), tau1_cor.item()/len(outliers_idx0), tau0_auc, tau1_auc] + scores_l 

'''
Ads data generate scores.
'''
def test_ads_data(opt):
    
    X, noise_idx = ads.get_data('data/internet_ads.arff') #('data/ads_05_nodup.arff')
    noise_idx = noise_idx.unsqueeze(-1)
    print('# of outliers {}'.format(noise_idx.size(0)))
    
    if True:
        X = X - X.mean(0)
    else:
        centers, codes = baselines.cluster(X, 5)
        #X1 = torch.zeros_like(X)
        for i in range(5):
            cur_X = X[codes==i]
            X[codes==i] = cur_X - cur_X.mean(0)            
    
    whiten = False
    if whiten:
        zeros = torch.zeros(len(X), device=utils.device)
        zeros[noise_idx] = 1        
        X_inliers = X[zeros==0]
        X_inliers = X_inliers[:-1:5] #len(X_inliers)/5]
        
        content_cov = utils.cov(X_inliers)
        U, D, V_t = linalg.svd(content_cov)
        cov_inv = torch.from_numpy(np.matmul(linalg.pinv(np.diag(np.sqrt(D)), rcond=1e-3), U.transpose())).to(utils.device)
        X = torch.mm(cov_inv, X.t()).t()        
        
    X0 = X
    if opt.fast_jl:
        X = utils.pad_to_2power(X)
    
    opt.n, opt.feat_dim = X.size(0), X.size(1)
    #percentage of points to remove
    opt.remove_p = 0.1
    opt.lamb_multiplier = 4
    #number of top dirs for calculating tau0
    opt.n_top_dir = 1
    opt.n_iter = 1
    
    print('samples size {} {}'.format(opt.n, opt.feat_dim))    
    tau1, select_idx1, n_removed1, tau0, select_idx0, n_removed0 = compute_tau1_tau0(X, opt)

    all_idx = torch.zeros(len(tau1), device=utils.device)
    all_idx[noise_idx] = 1
    outlier_method_l = ['knn'] #march
    outlier_method_l = ['l2', 'iso forest', 'lof', 'knn']
    
    scores_l = []
    opt.baseline = 'knn'#'l2' #'l2' #'tau0' #'l2'#'isolation_forest'#'dbscan' #'isolation_forest'
    for method in outlier_method_l:
        if method == 'iso forest':
            tau = baselines.isolation_forest(X)
        elif method == 'lof':
            tau = baselines.knn_dist_lof(X)
        elif method == 'ell env':
            tau = baselines.ellenv(X)
        elif method == 'dbscan':
            tau = baselines.dbscan(X)
        elif method == 'l2':
            tau = baselines.l2(X)
        elif method == 'knn':
            tau = baselines.knn_dist(X)        
        elif method == 'tau2':
            select_idx2 = torch.LongTensor(list(range(len(X)))).to(utils.device)
            tau = compute_tau2(X, select_idx2, opt)
        else:
            raise Exception('method {} not supported'.format(method))

        good_tau = tau[all_idx==0]
        bad_tau = tau[all_idx==1]
        auc = utils.auc(good_tau, bad_tau)
        scores_l.append(auc)
        #visualize_scores = True
        if opt.visualize_scores:
            pdb.set_trace()
            utils.inlier_outlier_hist(good_tau, bad_tau, method+'ads', high=20)            
        
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
        if True:
            #complement of noise_idx
            #X_range = list(range(len(X)))
            
            inliers_tau1 = tau1[all_idx==0] 
            outliers_tau1 = tau1[all_idx==1]
            #utils.inlier_outlier_hist(inliers_tau1, outliers_tau1, 'tau1_ads', high=40)
            
            inliers_tau0 = tau0[all_idx==0] 
            outliers_tau0 = tau0[all_idx==1]
            #utils.inlier_outlier_hist(inliers_tau0, outliers_tau0, opt.baseline+'_ads', high=40)
            
            #compute evals before and after adding outliers
            inliers = X0[all_idx==0]
            U, D_in, V_t = linalg.svd(inliers.cpu().numpy())

            _, D, _ = linalg.svd(X0.cpu().numpy())                        

    print('tau1 size {}'.format(tau1.size(0)))
    outliers_idx0_exp = outliers_idx0.unsqueeze(0).expand(len(noise_idx), -1)
    outliers_idx1_exp = outliers_idx1.unsqueeze(0).expand(len(noise_idx), -1)
    tau0_cor = noise_idx.eq(outliers_idx0_exp).sum()
    tau1_cor = noise_idx.eq(outliers_idx1_exp).sum()
    print('{}_cor {} out of {} tau1_cor {} out of {}'.format(opt.baseline, tau0_cor, len(outliers_idx0), tau1_cor, len(outliers_idx1)))
    auc1 = utils.auc(inliers_tau1, outliers_tau1)
    auc0 = utils.auc(inliers_tau0, outliers_tau0)
    print('auc0 {} auc1 {} '.format(auc0, auc1))
    print('others: {}'.format(scores_l))
    pdb.set_trace()
    
def test_vgg_data():
    
    #X = data.process_vgg_data()
    #the trailing denotes the number of layers peeled at the end of the net.
    X = torch.load('data/val_embs0.pt').to(utils.device)
    cor_bool = torch.load('data/cor_idx2.pt').to(utils.device) #whether prediction is correct
    
    #whether to test only one class.
    one_class = False
    sep_mean = True #take mean of classes separately
    sep_mean_sep_class = False
    #the splits in data (X) to run one round of removal on
    eval_split_l = [X.size(0)]
    
    if sep_mean:
        ##X_classes = torch.load('data/val_classes.pt').to(utils.device)
        X_classes = torch.load('data/pred_cls.pt').to(utils.device)
        all_idx = torch.LongTensor(range(len(X_classes))).to(utils.device)
        X1 = torch.zeros_like(X)
        means_l = []
        for i in range(10):
            cur_idx = all_idx[X_classes==i]
            
            cur_idx = cur_idx.unsqueeze(-1).expand(-1, X.size(-1))
            cur_X = X[X_classes==i]
            print('curX min {}'.format((cur_X.mean(0)**2).sum()))
            means_l.append(cur_X.mean(0))
            cur_X = cur_X - cur_X.mean(0)
            X1.scatter_(dim=0, index=cur_idx, src=cur_X)
        
        X = X1
        means = torch.stack(means_l, dim=0)
        
        mean_dist = baselines.dist(means, means)
        pdb.set_trace()
        #cor_bool = cor_bool[X_classes==select_class]
    elif sep_mean_sep_class:
        
        X_classes = torch.load('data/pred_cls.pt').to(utils.device)
        all_idx = torch.LongTensor(range(len(X_classes))).to(utils.device)
        X1 = torch.zeros_like(X)
        cor_bool1 = torch.zeros_like(cor_bool)
        eval_split_l = []
        cur_count = 0
        
        for i in range(10):
            cur_idx = all_idx[X_classes==i]
            cur_len = len(cur_idx)
            cur_X = X[X_classes==i]
            X1[cur_count : cur_count+cur_len] = cur_X - cur_X.mean(0)
            cor_bool1[cur_count : cur_count+cur_len] = cor_bool[X_classes==i]
            eval_split_l.append(cur_count+cur_len)
            cur_count += cur_len
        X = X1        
        cor_bool = cor_bool1
    elif one_class:
        #val_classes.pt contains ground truth class labels.
        #X_classes = torch.load('data/val_classes.pt').to(utils.device)
        X_classes = torch.load('data/pred_cls.pt').to(utils.device)
        select_class = 4
        X = X[X_classes==select_class]
        X = X - X.mean(0)
        cor_bool = cor_bool[X_classes==select_class]
    else:        
        X = X - X.mean(0)
        
    X = utils.pad_to_2power(X)
    
    opt = utils.parse_args()    
    opt.n, opt.feat_dim = X.size(0), X.size(1)
    #percentage of points to remove
    opt.remove_p = 0.1
    
    #number of top dirs for calculating tau0
    opt.n_top_dir = 1
    
    print('samples size {} {}'.format(opt.n, opt.feat_dim))
    opt.n_iter = 1
    l2_baseline_bool = True
    
    tau1_l, select_idx1_l, n_removed1, tau0_l, select_idx0_l, n_removed0 = [], [], 0, [], [], 0
    outliers_idx0_l = []
    outliers_idx1_l = []
    prev_idx = 0
    for i in eval_split_l:
        cur_X = X[prev_idx:i]
        
        opt.n = cur_X.size(0) #currently opt.n not used downstream
        tau1, select_idx1, cur_n_removed1, tau0, select_idx0, cur_n_removed0 = compute_tau1_tau0(cur_X, opt)
                    
        tau1_l.append(tau1)
        select_idx1_l.append(select_idx1+prev_idx)
        n_removed1 += cur_n_removed1
        if l2_baseline_bool:
            #all data are centered per class
            tau1 = (cur_X**2).sum(dim=-1)
        else:
            select_idx0_l.append(select_idx0+prev_idx)
            
        tau0_l.append(tau0)        
        n_removed0 += cur_n_removed0        
        
        outliers_idx0_l.append(prev_idx+torch.topk(tau0, k=cur_n_removed0, largest=True)[1])
        outliers_idx1_l.append(prev_idx+torch.topk(tau1, k=cur_n_removed1, largest=True)[1])
        prev_idx = i        
        
    tau1 = torch.cat(tau1_l, dim=0)
    #select_idx1 = torch.cat(select_idx1_l, dim=0)
    tau0 = torch.cat(tau0_l, dim=0)
    #select_idx0 = torch.cat(select_idx0_l, dim=0)

    outliers_idx0 = torch.cat(outliers_idx0_l, dim=0)
    outliers_idx1 = torch.cat(outliers_idx1_l, dim=0)
    
    #select where cor_idx is 0, indices where model made mistake
    if utils.device == 'cuda':
        wrong_idx = torch.cuda.LongTensor(range(cor_bool.size(0)))[cor_bool==0].unsqueeze(-1)
    else:
        wrong_idx = torch.LongTensor(range(cor_bool.size(0)))[cor_bool==0].unsqueeze(-1)
        
    pdb.set_trace()    
    outliers_idx1_exp = outliers_idx1.unsqueeze(0).expand(wrong_idx.size(0), -1)
    outliers_idx0_exp = outliers_idx0.unsqueeze(0).expand(wrong_idx.size(0), -1)
    #how many of the wrong indices were predicted by tau1 vs tau0
    tau1_cor = wrong_idx.eq(outliers_idx1_exp).sum()
    tau0_cor = wrong_idx.eq(outliers_idx0_exp).sum()
    print('tau1 cor {} tau0 cor {} n_removed {}'.format(tau1_cor, tau0_cor, n_removed1))
    pdb.set_trace()

    
if __name__=='__main__':

    opt = utils.parse_args()
    
    generate_data = True #False 
    opt.fast_jl = True #False 
    opt.use_std = True
    opt.compute_scores_diff = True
    opt.visualize_scores = False
    opt.whiten = True
    opt.high_dim = True #False 
    
    if generate_data:
        if opt.high_dim:
            utils.device = 'cpu'
            device = 'cpu'
            
        #glove, vgg, genetics, or syn
        dataset_name = 'syn'
        opt.dir = 'syn'
        opt.type = 'lamb'
        print('{} {}'.format(opt.dir, opt.type))
        if opt.type == 'lamb':
            generate_and_score_lamb(opt, dataset_name)
        elif opt.type == 'dirs':
            generate_and_score(opt, dataset_name)            
    else:
        dataset_name = 'glove_dirs' #'glove_lamb' #'glove_lamb' #'glove_dirs' #'glove'
        if dataset_name == 'glove':
            opt.dir = 'text'
            opt.type = '_'
            test_glove_data(opt)
        elif dataset_name == 'glove_lamb':
            opt.dir = 'text'
            opt.type = 'lamb'   
            test_glove_data_lamb(opt)
        elif dataset_name == 'glove_dirs':
            opt.dir = 'text'
            opt.type = 'dirs'   
            test_glove_data_dirs(opt)
        elif dataset_name == 'ads':
            test_ads_data(opt)
        elif dataset_name == 'genetics':
            test_genetics_data()
        elif dataset_name == 'vgg':
            test_vgg_data()
        else:
            generate_and_score(opt)


