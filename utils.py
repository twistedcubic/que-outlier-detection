
'''
Utilities functions
'''
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import os.path as osp
import utils
import argparse
import torch
import math
import numpy.linalg as linalg
#can be obtained from https://github.com/FALCONN-LIB/FFHT
import ffht
import scipy.linalg

import pdb

res_dir = 'results'
data_dir = 'data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_dir', default=1, type=int, help='Max number of directions' )
    parser.add_argument('--lamb_multiplier', type=float, default=1., help='Set alpha multiplier')
    parser.add_argument('--experiment_type', default='syn_lamb', help='Set type of experiment, e.g. syn_dirs, syn_lamb, text_lamb, text_dirs, image_lamb, image_dirs, representing varying alpha or the number of corruption directions for the respective dataset')
    parser.add_argument('--generate_data', help='Generate synthetic data to run synthetic data experiments', dest='generate_data', action='store_true')
    parser.set_defaults(generate_data=False)
    parser.add_argument('--fast_jl', help='Use fast method to generate approximate QUE scores', dest='fast_jl', action='store_true')
    parser.set_defaults(fast_jl=False)
    parser.add_argument('--fast_whiten', help='Use approximate whitening', dest='fast_whiten', action='store_true')
    parser.set_defaults(fast_whiten=False)    
    parser.add_argument('--high_dim', help='Generate high-dimensional data, if running synthetic data experiments', dest='high_dim', action='store_true')
    parser.set_defaults(high_dim=False)    
    
    opt = parser.parse_args()
    
    if len(opt.experiment_type) > 3 and opt.experiment_type[:3] == 'syn':
        opt.generate_data = True
        
    return opt

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
create_dir(res_dir)

'''
Get degree and coefficient for kth Chebyshev poly.
'''
def get_chebyshev_deg(k):
    if k == 0:
        coeff = [1]
        deg = [0]   
    elif k == 1:
        coeff = [1]
        deg = [1]
    elif k == 2:
        coeff = [2, -1]
        deg = [2, 0]
    elif k == 3:
        coeff = [4, -3]
        deg = [3, 1]
    elif k == 4:
        coeff = [8, -8, 1]
        deg = [4, 2, 0]
    elif k == 5:
        coeff = [16, -20, 5]
        deg = [5, 3, 1]
    elif k == 6:
        coeff = [32, -48, 18, -1]
        deg = [6, 4, 2, 0]
    else:
        raise Exception('deg {} chebyshev not supported'.format(k))
    return coeff, deg
        
'''
Combination of JL projection and 
Chebyshev expansion of the matrix exponential.
Input:
-X: data matrix, 2D tensor. X is sparse for gene data!
Returns:
-tau1: scores, 1D tensor (n,)
'''
def jl_chebyshev(X, lamb):

    #print(X[:,0].mean(0))
    #assert X[:,0].mean(0) < 1e-4
    X = X - X.mean(0, keepdim=True)
    
    n_data, feat_dim = X.size()
    X_scaled = X/n_data
    
    #if lamb=0 no scaling, so not to magnify bessel[i, 0] in the approximation.
    scale = int(dominant_eval_cov(np.sqrt(lamb)*X)[0]) if lamb > 0 else 1
    
    if scale > 1:
        print('Scaling M! {}'.format(scale))
        #scale down matrix if matrix norm >= 3, since later scale up when
        #odd power
        if scale%2 == 0:
            scale -= 1
        X_scaled /= scale
    else:
        scale = 1    
    #k = int(math.log(feat_dim, 2))
    subsample_freq = int(feat_dim/math.log(feat_dim, 2)) #100
    k = math.ceil(feat_dim/subsample_freq)

    X_t = X.t() 
    #fast Hadamard transform (ffht) vs transform by multiplication by Hadamard mx.
    ffht_b = False 
    P, H, D = get_jl_mx(feat_dim, k, ffht_b)
    
    I_proj = torch.eye(feat_dim, feat_dim, device=X.device)
    
    M = D
    I_proj = torch.mm(D, I_proj)
    if ffht_b:
        
        M = M.t()
        #M1 = np.zeros((M.size(0), M.size(1)), dtype=np.double)
        M_np = M.cpu().numpy()
        
        I_np = I_proj.cpu().numpy()
        for i in range(M.size(0)):
            ffht.fht(M_np[i])
        for i in range(I_proj.size(0)):
            ffht.fht(I_np[i])
        #pdb.set_trace()
        M = torch.from_numpy(M_np).to(dtype=M.dtype, device=X.device).t()
        I_proj = torch.from_numpy(I_np).to(dtype=M.dtype, device=X.device)
    else:
        #right now form the matrix exponential here
        M = torch.mm(H, M)
        I_proj = torch.mm(H, I_proj)
            
    #apply P now so downstream multiplications are faster: kd instead of d^2.
    #subsample to get reduced dimension
    subsample = True
    if subsample:
        #random sampling performs well in practice and has lower complexity        
        #select_idx = torch.randint(low=0, high=feat_dim, size=(feat_dim//5,)) <--this produces repeats
        if device == 'cuda':
            #pdb.set_trace()
            select_idx = torch.cuda.LongTensor(list(range(0, feat_dim, subsample_freq)))
        else:
            select_idx = torch.LongTensor(list(range(0, feat_dim, subsample_freq)))
        #M = torch.index_select(M, dim=0, index=select_idx)
        M = M[select_idx]
        #I_proj = torch.index_select(I_proj, dim=0, index=select_idx)
        I_proj = I_proj[select_idx]
    else:
        M = torch.sparse.mm(P, M)
        I_proj = torch.sparse.mm(P, I_proj)

    #M is now the projection mx
    A = M
    for _ in range(scale):
        #(k x d)
        A = sketch_and_apply(lamb, X, X_scaled, A, I_proj)
        
    #Compute tau1 scores
    #M = M / M.diag().sum()
    #M is (k x d)
    #compute tau1 scores (this M is previous M^{1/2})
    tau1 = (torch.mm(A, X_t)**2).sum(0)
    
    return tau1

'''
-M: projection mx
-X, X_scaled, input and scaled input
Returns:
-k x d projected matrix
'''
def sketch_and_apply(lamb, X, X_scaled, M, I_proj):
    X_t = X.t()
    M = torch.mm(M, X_t)
    M = torch.mm(M, X_scaled)
    
    check_cov = False
    if check_cov:
        #sanity check, use exact cov mx
        #print('Using real cov mx!')        
        M = cov(X)
        subsample_freq = 1
        feat_dim = X.size(1)
        k = feat_dim
        I_proj = torch.eye(k, k, device=X.device)

    check_exp = False
    #Sanity check, computes exact matrix expoenntial
    if False:
        U, D, V_t = linalg.svd(lamb*M.cpu().numpy())
        pdb.set_trace()
        U = torch.from_numpy(U.astype('float32')).to(device)
        D_exp = torch.from_numpy(np.exp(D.astype('float32'))).to(device).diag()
        m = torch.mm(U, D_exp)
        m = torch.mm(m, U.t())        
        #tau1 = (torch.mm(M, X_t)**2).sum(0)        
        return m
    if check_exp:
        M = torch.from_numpy(scipy.linalg.expm(lamb*M.cpu().numpy())).to(device)
        #pdb.set_trace()
        tau1 = (torch.mm(M, X_t)**2).sum(0)
        #X_m = torch.mm(X, M)
        #tau1 = (X*X_m).sum(-1)
        return M
    
    ## Matrix exponential appx ##
    total_deg = 6
    monomials = [0]*total_deg
    #k x d
    monomials[1] = M
    
    #create monomimials in chebyshev poly. Start with deg 2 since already multiplied with one cov.
    for i in range(2, total_deg):        
        monomials[i] = torch.mm(torch.mm(monomials[i-1], X_t), X_scaled)
    
    monomials[0] = I_proj 
    M = 0
    #M is now (k x d)
    #degrees of terms in deg^th chebyshev poly
    for kk in range(1, total_deg):
        #coefficients and degrees for chebyshev poly. Includes 0th deg.  
        coeff, deg = get_chebyshev_deg(kk)

        T_k = 0
        for i, d in enumerate(deg):
            c = coeff[i]            
            T_k += c*lamb**d*monomials[d]
            
        #includes multiplication with powers of i
        bessel_k = get_bessel('-i', kk)
        M = M + bessel_k*T_k

    #M = I_proj
    #degree 0 term. M is now (k x d)
    #M[:, :k] = 2*M[:, :k] + get_bessel('i', 0) * torch.eye(k, feat_dim, device=X.device) #torch.ones((k,), device=X.device).diag()
    #(k x d) matrix
    M = 2*M + get_bessel('i', 0) * I_proj 

    return M
    
    
'''
Create JL projection matrix.
Input: 
-d: original dim
-k: reduced dim
'''
def get_jl_mx(d, k, ffht_b):
    #M is sparse k x d matrix
    
    P = torch.ones(k, d, device=device) #torch.sparse(  )

    if not ffht_b:
        H = get_hadamard(d)
    else:
        H = None
    #diagonal Rademacher mx
    sign = torch.randint(low=0, high=2, size=(d,), device=device, dtype=torch.float32)
    sign[sign==0] = -1
    D = sign.diag()
    
    return P, H, D

#dict of Hadamard matrices of given dimensions
H2 = {}
'''
-d: dimension of H. Power of 2.
-replace with FFT for d log(d).
'''
def get_hadamard(d):

    if d in H2:
        return H2[d]
    if osp.exists('h{}.pt'.format(d)):
        H2[d] = torch.load('h{}.pt'.format(d)).to(device)
        return H2[d]
    power = math.log(d, 2)
    if power-round(power) != 0:
        raise Exception('Dimension of Hamadard matrix must be power of 2')
    power = int(power)
    #M1 = torch.FloatTensor([[ ], [ ]])
    M2 = torch.FloatTensor([1, 1, 1, -1])
    if device == 'cuda':
        M2 = M2.cuda()
    i = 2
    H = M2
    while i <= power:
        #H = torch.ger(H.view(-1), M2).view(2**i, 2**i)
        H = torch.ger(M2, H.view(-1))
        #reshape into 4 block matrices
        H = H.view(-1, 2**(i-1), 2**(i-1))
        H = torch.cat((torch.cat((H[0], H[1]), dim=1), torch.cat((H[2], H[3]), dim=1)), dim=0)
        #if i == 2:
        #    pdb.set_trace()
        i += 1
    torch.save(H, 'h{}.pt'.format(d))
    H2[d] = H.view(d, d) / np.sqrt(d)
    return H2[d]

'''
Pad to power of 2.
Input: size 2.
'''
def pad_to_2power(X):
    n_data, feat_dim = X.size(0), X.size(-1)
    power = int(math.ceil(math.log(feat_dim, 2)))
    power_diff = 2**power-feat_dim
    if power_diff == 0:
        return X
    padding = torch.zeros(n_data, power_diff, dtype=X.dtype, device=X.device)
    X = torch.cat((X, padding), dim=-1)
    
    return X

'''
Find dominant eval of XX^t (and evec in the process) using the power method.
Without explicitly forming XX^t
Returns:
-dominant eval + corresponding eigenvector
'''
def dominant_eval_cov(X):
    n_data = X.size(0)
    X = X - X.mean(dim=0, keepdim=True)
    X_t = X.t()
    X_t_scaled = X_t/n_data
    n_round = 5
    
    v = torch.randn(X.size(-1), 1, device=X.device)
    for _ in range(n_round):
        v = torch.mm(X_t_scaled, torch.mm(X, v))
        #scale each time instead of at the end to avoid overflow
        #v = v / (v**2).sum().sqrt()
    v = v / (v**2).sum().sqrt()
    mu = torch.mm(v.t(), torch.mm(X_t_scaled, torch.mm(X, v))) / (v**2).sum()
    
    return mu.item(), v.view(-1)
'''
dominant eval of matrix X
Returns: top eval and evec
'''
def dominant_eval(A):
    '''
    n_data = X.size(0)
    X = X - X.mean(dim=0, keepdim=True)
    X_t = X.t()
    X_t_scaled = X_t/n_data
    '''
    n_round = 5    
    v = torch.randn(A.size(-1), 1, device=A.device)
    for _ in range(n_round):
        v = torch.mm(A, v)
        #scale each time instead of at the end to avoid overflow
        #v = v / (v**2).sum().sqrt()
    v = v / (v**2).sum().sqrt()
    mu = torch.mm(v.t(), torch.mm(A, v)) / (v**2).sum()
    
    return mu.item(), v.view(-1)

'''
Top k eigenvalues of X_c X_c^t rather than top one.
'''
def dominant_eval_k(A, k):
    
    evals = torch.zeros(k).to(device)
    evecs = torch.zeros(k, A.size(-1)).to(device)
    
    for i in range(k):
        
        cur_eval, cur_evec = dominant_eval(A)
        A -= (cur_evec*A).sum(-1, keepdim=True) * (cur_evec/(cur_evec**2).sum())
        
        evals[i] = cur_eval
        evecs[i] = cur_evec
        
    return evals, evecs

'''
Top cov dir, for e.g. visualization + debugging.
'''
def get_top_evals(X, k=10):
    X_cov = cov(X)
    U, D, V_t = linalg.svd(X_cov.cpu().numpy())
    return D[:k]

#bessel function values at i and -i, index is degree.
#sum_{j=0}^\infty ((-1)^j/(2^(2j+k) *j!*(k+j)! )) * (-i)^(2*j+k) for k=0 BesselI(0, 1)
bessel_i = [1.1266066]
bessel_neg_i = [1.266066, -0.565159j, -0.1357476, 0.0221684j, 0.00273712, -0.00027146j]
#includes multipliacation with powers of i, i**k
bessel_neg_i = [1.266066, 0.565159, 0.1357476, 0.0221684, 0.00273712, 0.00027146]

'''
Get precomputed deg^th Bessel function value at input arg.
'''
def get_bessel(arg, deg):
    if arg == 'i':
        if deg > len(bessel_i):
            raise Exception('Bessel i not computed for deg {}'.format(deg))         
        return bessel_i[deg]
    elif arg == '-i':
        if deg > len(bessel_neg_i):
            raise Exception('Bessel -i not computed for deg {}'.format(deg))
        return bessel_neg_i[deg]


'''
Projection (vector) of dirs onto target direction.
'''
def project_onto(tgt, dirs):
    
    projection = (tgt*dirs).sum(-1, keepdims=True) * (tgt/(tgt**2).sum())
    
    return projection

'''
Plot 
legends: last field of which correponds to hue, and dictates which kind of plot, eg lambda or p
'''
#def plot_acc(acc_l, k_l, p_l, tau_l, opt):
def plot_acc(k_l, acc_l, tau_l, p_l, legends, opt):
    
    opt.lamb = round(opt.lamb, 2)
    df = create_df(k_l, acc_l, tau_l, p_l, legends)
    #fig = sns.scatterplot(x='k', y='acc', style='tau', hue='p', data=df)
    fig = sns.scatterplot(x=legends[0], y=legends[1], style=legends[2], hue=legends[3], data=df)
    
    fig.set(ylim=(0, 1.05))
    fig.set_title('acc vs k. n_iter {} remove_fac {} p {} on dataset {} tau0: {}'.format(opt.n_iter, opt.remove_factor, opt.p, opt.dataset_name, opt.baseline))
    fig_path = osp.join(utils.res_dir, 'plot{}{}_{}_{}_{}{}iter{}{}{}.jpg'.format('N{}_noise'.format(opt.noise_norm_div),opt.feat_dim, opt.n_dir, opt.norm_scale, legends[-1], opt.p, opt.n_iter, opt.dataset_name, opt.baseline))
    fig.figure.savefig(fig_path)
    print('figure saved under {}'.format(fig_path))

'''
Plot wrt lambda
'''
def plot_acc_syn_lamb(p_l, acc_l, tau_l, legends, opt):
    
    opt.lamb = round(opt.lamb, 2)
    df = pd.DataFrame({legends[0]:p_l, legends[1]:acc_l, legends[2]:tau_l})
    #df = create_df(p_l, acc_l, tau_l, legends)
    #fig = sns.scatterplot(x='k', y='acc', style='tau', hue='p', data=df)
    fig = sns.scatterplot(x=legends[0], y=legends[1], style=legends[2], data=df)
    
    fig.set(ylim=(0, 1.05))
    fig.set_title('acc vs k. n_iter {} remove_fac {} p {} on dataset {} tau0: {}'.format(opt.n_iter, opt.remove_factor, opt.p, opt.dataset_name, opt.baseline))
    fig_path = osp.join(utils.res_dir, 'syn', 'lamb_{}_{}_{}_{}{}iter{}{}{}.jpg'.format(opt.feat_dim, opt.n_dir, opt.norm_scale, legends[-1], opt.p, opt.n_iter, opt.dataset_name, opt.baseline))
    fig.figure.savefig(fig_path)
    print('figure saved under {}'.format(fig_path))

'''
Scatter plot of input X, e.g. for varying lambda
Input:
-standard deviation: standard deviation around each point.
'''
def plot_scatter(X, Y, legends, opt, std=None):
    df = pd.DataFrame({legends[0]:X, legends[1]:Y})
    
    fig = sns.scatterplot(x=legends[0], y=legends[1], data=df, label=(legends[1]))
    #fig = sns.scatterplot(x=legends[0], y=legends[1], style=legends[2], hue=legends[3], data=df)

    fig.set(ylim=(0, 1.05))
    plt.grid(True)
    #fig.set(ylim=(0, max(Y)+.1))
    #fig.set_title('acc vs k. n_iter {} remove_fac {} p {} noise_norm_div {} on dataset {} tau0: {}'.format(opt.n_iter, opt.remove_factor, opt.p, opt.noise_norm_div, opt.dataset_name, opt.baseline))
    fig.set_title('Recall scores as a function of varying {} for text {}'.format(legends[0], opt.text_name))
    fig_path = osp.join(utils.res_dir, 'text', '{}_{}_{}1.jpg'.format(legends[0], legends[1], opt.text_name))
    fig.figure.savefig(fig_path)
    print('figure saved under {}'.format(fig_path))

'''
Plot flexible number of variables.
Useful for e.g. plotting wrt baselines.
Input:
-data_l: 0th entry is the x axis.
Inputs are np arrays
-legend_l has len one less than data_l
-name: extra appendix to file name.
'''
def plot_scatter_flex(data_ar, legend_l, opt, std_ar=None, name=''):
    
    plt.clf()
    m = {}
    markers = ['^', 'o', 'x', '.', '1', '3', '+', '4', '5']
    '''
    legends = []
    for i, data in data_l:
        m[legend_l[i]] = data
    '''
    for i in range(1, len(data_ar)):
        #plt.scatter(data_ar[0], data_ar[i], marker=markers[i-1], label=legend_l[i-1])
        cur_legend = get_label_name(legend_l[i-1])
        plt.errorbar(data_ar[0], data_ar[i], yerr=std_ar[i], marker=markers[i-1], label=cur_legend) 
    
    label_name = get_label_name(name) #'naive spectral' if name == 'tau0' else name
    #fig.set(ylim=(0, 1.05))
    
    plt.grid(True)
    plt.legend()
    if opt.type == 'lamb':
        plt.xlabel('Alpha')
        plt.ylabel('ROCAUC(QUE) - ROCAUC{}'.format(label_name))
        plt.title('ROCAUC(QUE) improvement over ROCAUC({})'.format(label_name))
    else:
        x_label = get_label_name(opt.type)
        plt.xlabel(x_label)
        plt.ylabel('ROCAUC')
        plt.title('ROCAUC of QUE vs baseline methods')
    #fig.set_title('acc vs k. n_iter {} remove_fac {} p {} noise_norm_div {} on dataset {} tau0: {}'.format(opt.n_iter, opt.remove_factor, opt.p, opt.noise_norm_div, opt.dataset_name, opt.baseline))
    
    fname_append = ''
    if not opt.whiten:
        fname_append += '_nw'
    if opt.fast_whiten:
        fname_append += '_fw'
    if opt.fast_jl:
        fname_append += '_fast'
        
    #if opt.fast_jl:
    #    fig_path = osp.join(utils.res_dir, opt.dir, 'baselines_{}{}_fast.jpg'.format(opt.type, name))
    #else:
    create_dir(osp.join(utils.res_dir, opt.dir))
    fig_path = osp.join(utils.res_dir, opt.dir, 'baselines_{}{}{}.jpg'.format(opt.type, name, fname_append))
    plt.savefig(fig_path)
    print('figure saved under {}'.format(fig_path))

'''
Get label name to be used on plots.
'''
def get_label_name(name):
    name2label = {'tau0':'naive spectral', 'tau1':'QUE', 'lamb':'Alpha',
                  'dirs':'number of directions (k)'}
    try:
        return name2label[name]
    except KeyError:
        return name
    
'''
Computes average probability of outlier scores higher than inlier scores.
Input:
-inlier+outlier scores, 1D tensors
'''
def auc(inlier_scores, outlier_scores0):
    
    n_inliers, n_outliers = len(inlier_scores), len(outlier_scores0)
    if False and n_inliers + n_outliers > 150000:
        inlier_scores = inlier_scores.to('cpu')
        outlier_scores = outlier_scores0.to('cpu')
    prob_l = []
    chunk_sz = 500
    for i in range(0, n_outliers, chunk_sz):
        start = i
        end = min(n_outliers, i+chunk_sz)
        cur_n = end - start
        outlier_scores = outlier_scores0[start:end]
        
        #average probabilities of inliers scores lower than outlier scores.    
        outlier_scores_exp = outlier_scores.unsqueeze(-1).expand(-1, n_inliers)
        inlier_scores_exp = inlier_scores.unsqueeze(0).expand(cur_n, -1)
        zeros = torch.zeros(cur_n, n_inliers).to(device)
        zeros[outlier_scores_exp > inlier_scores_exp] = 1
        prob = (zeros.sum(-1) / n_inliers).mean().item()
        prob_l.append(prob)
    return np.mean(prob_l)
    
'''
Plot histogram of tensors
Input:
-data
-keyword to be used in file
'''
def hist(X, name, high=10):
    X = X.cpu().numpy()
    plt.hist(X, 50, label=str(name))
    
    plt.xlabel('projection')
    plt.ylabel('count')
    plt.title('projections of {} onto top covariance dir'.format(name))
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    
    plt.axis([X.min(), X.max(), 0, high])
    plt.grid(True)
    
    fig_path = osp.join(utils.res_dir, 'eval_proj_{}.jpg'.format(name))
    plt.savefig(fig_path)
    print('figure saved under {}'.format(fig_path))

'''
Inliers and outliers histograms.
Input:
-X/Y: inliers/outliers score (or other measurement) distributions according to some score
-
'''
def inlier_outlier_hist(X, Y, score_name, high=50):
    #X, Y
    
    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    n_bins_x = 50
    n_bins_y = max(1, int(n_bins_x * (Y.max()-Y.min()) / (X.max()-X.min())))
    plt.hist(X, n_bins_x, label='inliers')
    plt.hist(Y, n_bins_y, label='outliers')
    
    plt.xlabel('knn distance')
    plt.ylabel('sample count')
    label_name = get_label_name(score_name)
    plt.title('Distance to k-nearest neighbors'.format(label_name))
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')

    plt.legend()
    #plt.axis([min(X.min(), Y.min()), max(X.max(), Y.max()), 0, high])
    plt.axis([min(X.min(), Y.min()), 30, 0, high]) #for ads,high y 300 
    #plt.axis([min(X.min(), Y.min()), 3, 0, high]) #syn high x 3
    plt.grid(True)
    
    fig_path = osp.join(utils.res_dir, 'knn_inout_{}.jpg'.format(score_name))
    plt.savefig(fig_path)
    print('figure saved under {}'.format(fig_path))

    
'''
k is number of dirs
p_mx is percentage of all noise combined.
legends: array of strings of legends, eg ['a', 'b', 'c', 'd']
'''
def create_df(k_l, acc_l, tau_l, p_l, legends):

    #return pd.DataFrame({'acc':acc_l, 'k':k_l, 'tau':tau_l, 'p':p_l})
    return pd.DataFrame({legends[0]:k_l, legends[1]:acc_l, legends[2]:tau_l, legends[3]:p_l})

'''
Take inner product of rows in one with rows in another.
Input:
-2D tensors
'''
def inner(mx1, mx2):
    return (mx1 * mx2).sum(dim=1)

'''
Inner product matrix of all pairwise rows and columns
'''
def inner_mx(mx1, mx2):    
    return torch.mm(mx1 * mx2.t())


'''
Input: lines is list of objects, not newline-terminated yet. 
'''
def write_lines(lines, path, mode='w'):
    lines1 = []
    for line in lines:
        lines1.append(str(line) + os.linesep)
    with open(path, mode) as file:
        file.writelines(lines1)
        
def read_lines(path):
    with open(path, 'r') as file:
        return file.readlines()
    
'''
Input:
-X: shape (n_sample, n_feat)
'''
def cov(X):
    #X_mean = X.mean()
    X = X - X.mean(dim=0, keepdim=True)

    cov = torch.mm(X.t(), X) / X.size(0)
    return cov
    
########################

def create_df_(acc_mx, probe_mx, height, k, opt):
    #construct probe_count, acc, and dist_count                                                                                  
    #total number of points we compute distances to                                                                                  dist_count_l = []
    acc_l = []
    probe_l = []
    counter = 0
    n_clusters_ar = [2**(i+1) for i in range(20)]

    #i indicates n_clusters
    for i, acc_ar in enumerate(acc_mx):
        n_clusters = n_clusters_ar[i]
        #j is n_bins
        for j, acc in enumerate(acc_ar):
            probe_count = probe_mx[i][j]
            if not opt.glove and not opt.sift:
                if height == 1 and probe_count > 2000:
                    continue
                elif probe_count > 3000:
                    continue

            # \sum_u^h n_bins^u * n_clusters * k
            exp = np.array([l for l in range(height)])
            
            dist_count = np.sum(k * n_clusters * j**exp)
            if not opt.glove and not opt.sift:
                if dist_count > 50000:
                    continue
            dist_count_l.append(dist_count)
            acc_l.append(acc)
            #probe_l.append(probe_count)
            probe_l.append(probe_count + dist_count)

            counter += 1
            
    df = pd.DataFrame({'probe_count':probe_l, 'acc':acc_l, 'dist_count':dist_count_l})
    return df

def plot_acc_():    
    df_l = []
    height_df_l = []
    for i, acc_mx in enumerate(acc_mx_l):
        probe_mx = probe_mx_l[i]
        height = height_l[i]
        df = create_df(acc_mx, probe_mx, height, k, opt)
        df_l.append(df)
        height_df_l.extend([height] * len(df))

    method, max_loyd = json_data['km_method'], json_data['max_loyd']
    df = pd.concat(df_l, axis=0, ignore_index=True)

    height_df = pd.DataFrame({'height': height_df_l})

    df = pd.concat([df, height_df], axis=1)

    fig = sns.scatterplot(x='probe_count', y='acc', hue='height', data=df)
    
    fig.set_title('')
    fig_path = osp.join(' ', ' ')
    fig.figure.savefig(fig_path)

def np_save(obj, path):
    with open(path, 'wb') as f:
        np.save(f, obj)
        print('saved under {}'.format(path))

'''
Memory-compatible. 
Ranks of closest points not self.
Uses l2 dist. But uses cosine dist if data normalized. 
Input: 
-data: tensors
-specify k if only interested in the top k results.
-largest: whether pick largest when ranking. 
-include_self: include the point itself in the final ranking.
'''
def dist_rank(data_x, k, data_y=None, largest=False, opt=None, include_self=False):

    if isinstance(data_x, np.ndarray):
        data_x = torch.from_numpy(data_x)

    if data_y is None:
        data_y = data_x
    else:
        if isinstance(data_y, np.ndarray):
            data_y = torch.from_numpy(data_y)
    k0 = k
    device_o = data_x.device
    data_x = data_x.to(device)
    data_y = data_y.to(device)
    
    (data_x_len, dim) = data_x.size()
    data_y_len = data_y.size(0)
    #break into chunks. 5e6  is total for MNIST point size
    #chunk_sz = int(5e6 // data_y_len)
    chunk_sz = 16384
    chunk_sz = 500 #700 mem error. 1 mil points
    if data_y_len > 990000:
        chunk_sz = 600 #1000 if over 1.1 mil
        #chunk_sz = 500 #1000 if over 1.1 mil 
    else:
        chunk_sz = 3000    

    if k+1 > len(data_y):
        k = len(data_y) - 1
    #if opt is not None and opt.sift:
    
    if device == 'cuda':
        dist_mx = torch.cuda.LongTensor(data_x_len, k+1)
        act_dist = torch.cuda.FloatTensor(data_x_len, k+1)
    else:
        dist_mx = torch.LongTensor(data_x_len, k+1)
        act_dist = torch.cuda.FloatTensor(data_x_len, k+1)
    data_normalized = True if opt is not None and opt.normalize_data else False
    largest = True if largest else (True if data_normalized else False)
    
    #compute l2 dist <--be memory efficient by blocking
    total_chunks = int((data_x_len-1) // chunk_sz) + 1
    y_t = data_y.t()
    if not data_normalized:
        y_norm = (data_y**2).sum(-1).view(1, -1)
    
    for i in range(total_chunks):
        base = i*chunk_sz
        upto = min((i+1)*chunk_sz, data_x_len)
        cur_len = upto-base
        x = data_x[base : upto]
        
        if not data_normalized:
            x_norm = (x**2).sum(-1).view(-1, 1)        
            #plus op broadcasts
            dist = x_norm + y_norm        
            dist -= 2*torch.mm(x, y_t)
        else:
            dist = -torch.mm(x, y_t)
            
        topk_d, topk = torch.topk(dist, k=k+1, dim=1, largest=largest)
                
        dist_mx[base:upto, :k+1] = topk #torch.topk(dist, k=k+1, dim=1, largest=largest)[1][:, 1:]
        act_dist[base:upto, :k+1] = topk_d #torch.topk(dist, k=k+1, dim=1, largest=largest)[1][:, 1:]
        
    topk = dist_mx
    if k > 3 and opt is not None and opt.sift:
        #topk = dist_mx
        #sift contains duplicate points, don't run this in general.
        identity_ranks = torch.LongTensor(range(len(topk))).to(topk.device)
        topk_0 = topk[:, 0]
        topk_1 = topk[:, 1]
        topk_2 = topk[:, 2]
        topk_3 = topk[:, 3]

        id_idx1 = topk_1 == identity_ranks
        id_idx2 = topk_2 == identity_ranks
        id_idx3 = topk_3 == identity_ranks

        if torch.sum(id_idx1).item() > 0:
            topk[id_idx1, 1] = topk_0[id_idx1]

        if torch.sum(id_idx2).item() > 0:
            topk[id_idx2, 2] = topk_0[id_idx2]

        if torch.sum(id_idx3).item() > 0:
            topk[id_idx3, 3] = topk_0[id_idx3]           

    
    if not include_self:
        topk = topk[:, 1:]
        act_dist = act_dist[:, 1:]
    elif topk.size(-1) > k0:
        topk = topk[:, :-1]
    topk = topk.to(device_o)
    return act_dist, topk
