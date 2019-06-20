
'''
Baseline methods.
-kmeans clustering
-LOF
-isolation forest
-dbscan
-
'''
import torch
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.covariance
import sklearn.cluster
import random
import part_utils

import utils
import pdb

'''
kNN method that uses distances to k nearest neighbors
as scores. (global method)
Input:
-X: data, 2D tensor.
'''
def knn_dist(X, k=10, sum_dist=False):
    
    #dist_ = dist(X, X)
    #min_dist, idx = torch.topk(dist_, dim=-1, k=k, largest=False)
    min_dist, idx = part_utils.dist_rank(X, k=k, largest=False)
    
    if sum_dist:
        dist_score = min_dist.sum(-1)
    else:
        dist_score = min_dist.mean(-1)
    
    return dist_score

'''
Lof method using reachability criteria to determine density.
(Local method.)
'''
def knn_dist_lof(X, k=10):
    X_len = len(X)
    
    #dist_ = dist(X, X)    
    #min_dist, min_idx = torch.topk(dist_, dim=-1, k=k, largest=False)
    
    min_dist, min_idx = part_utils.dist_rank(X, k=k, largest=False)
    #if X_len == 8049:
    #    pdb.set_trace()
    kth_dist = min_dist[:, -1]
    # sum max(kth dist, dist(o, p)) over neighbors o of p
    kth_dist_exp = kth_dist.expand(X.size(0), -1) #n x n
    kth_dist = torch.gather(input=kth_dist_exp, dim=1, index=min_idx)
    
    try:
        min_dist[kth_dist > min_dist] = kth_dist[kth_dist > min_dist]
    except RuntimeError:
        print('erro!')
        pdb.set_trace()
    #inverse of lrd scores
    dist_avg = min_dist.mean(-1).clamp(min=0.0001)
    
    compare_density = False
    if compare_density:
        #compare with density. Get kth neighbor index.
        dist_avg_exp = dist_avg.unsqueeze(-1) / dist_avg.unsqueeze(0).expand(X_len, -1)
        #lof = torch.zeros(X_len, 1).to(utils.device)
        lof = torch.gather(input=dist_avg_exp, dim=-1, index=min_idx).sum(-1)
        torch.scatter_add_(lof, dim=-1, index=min_idx, src=dist_avg_exp)    
        return -lof.squeeze(0)

    return dist_avg

'''
LoOP: kNN based method using quadratic mean distance to estimate density.
LoOP (Local Outlier Probabilities) (Kriegel et al. 2009a)
'''
def knn_dist_loop(X, k=10):
    dist_ = dist(X, X)
    min_dist, idx = torch.topk(dist_, dim=-1, k=k, largest=False)
    dist_avg = (min_dist**2).mean(-1).sqrt()
    
    return dist_avg

'''
Isolation forest to compute outlier scores.
Returns: The higher the score, the more likely to be outlier.
'''
def isolation_forest(X):
    X = X.cpu().numpy()
    model = sklearn.ensemble.IsolationForest(contamination='auto', behaviour='new')
    #labels = model.fit_predict(X)
    model.fit(X)
    scores = -model.decision_function(X)
    
    #labels = torch.from_numpy(labels).to(utils.device)  
    #scores = torch.zeros_like(labels)
    #scores[labels==-1] = 1
    return torch.from_numpy(scores).to(utils.device)

'''
Elliptic envelope
Returns: The higher the score, the more likely to be outlier.
'''
def ellenv(X):
    X = X.cpu().numpy()
    model = sklearn.covariance.EllipticEnvelope(contamination=0.2)
    #ensemble.IsolationForest(contamination='auto', behaviour='new')
    model.fit(X)
    scores = -model.decision_function(X)
    
    #labels = torch.from_numpy(labels).to(utils.device)  
    #scores = torch.zeros_like(labels)
    #scores[labels==-1] = 1
    return torch.from_numpy(scores).to(utils.device)

'''
Local outlier factor.

'''
def lof(X):
    #precompute distances to accelerate LOF
    dist_mx = dist(X, X)    
    dist_mx = dist_mx.cpu().numpy()
    #metric by default is minkowski with p=2
    model = sklearn.neighbors.LocalOutlierFactor(n_neighbors=20, metric='precomputed', contamination='auto')
    labels = model.fit_predict(dist_mx)
    labels = torch.from_numpy(labels).to(utils.device)
    scores = torch.zeros_like(labels)
    scores[labels==-1] = 1
    return scores
    
    
'''
DBSCAN, density based, mark points as inlier if 
they either have lots of neighbors or have inliers
as their neighbors.
-X are points, not pairwise distances.
Returns:
-scores, 1 means outlier.
'''
def dbscan(X):
    X = X.cpu().numpy()
    model = sklearn.cluster.DBSCAN(min_samples=10)
    model.fit(X)
    #-1 means "outlier"
    labels = model.labels_
    labels = torch.from_numpy(labels).to(utils.device)
    scores = torch.zeros_like(labels)
    scores[labels==-1] = 1
    return scores

'''
Compute score using l2 distance to the mean
Higher scores mean more likely outliers.
'''
def l2(X):
    scores = ((X - X.mean(0))**2).sum(-1)    
    return scores
    
'''
Input:
-X, Y: 2D tensors
'''
def dist(X, Y):
    
    X_norms = torch.sum(X**2, dim=1).view(-1, 1)
    Y_norms = torch.sum(Y**2, dim=1).view(1, -1)
    cur_distances = X_norms + Y_norms - 2*torch.mm(X, Y.t())    

    return cur_distances
    
#number of iterations in 
MAX_KM_ITER = 60

def random_init(dataset, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    used = torch.zeros(num_points, dtype=torch.long)
    indices = torch.zeros(num_centers, dtype=torch.long)
    for i in range(num_centers):
        while True:
            cur_id = random.randint(0, num_points - 1)
            if used[cur_id] > 0:
                continue
            used[cur_id] = 1
            indices[i] = cur_id
            break
    indices = indices.to(dataset.device)
    centers = torch.gather(dataset, 0, indices.view(-1, 1).expand(-1, dimension))
    return centers

def compute_codes(dataset, centers):

    device0 = dataset.device
    dataset = dataset.cuda()
    centers = centers.cuda()

    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)

    chunk_size = int(8e8 / num_centers) 
    codes = torch.zeros(num_points, dtype=torch.long, device=utils.device)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers ** 2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
    del(dataset)
    del(centers)
    return codes.to(device0)

def update_centers(dataset, codes, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    cur_device = dataset.device
    centers = torch.zeros(num_centers, dimension, dtype=torch.float, device=cur_device)
    cnt = torch.zeros(num_centers, dtype=torch.float, device=cur_device)
    centers.scatter_add_(0, codes.view(-1, 1).expand(-1, dimension), dataset)
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float, device=cur_device))
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float, device=cur_device))
    centers /= cnt.view(-1, 1)
    return centers

'''
Clustering, but this requires number of clusters to be passed.
'''
def cluster(dataset, num_centers):
    centers = random_init(dataset, num_centers)
    codes = compute_codes(dataset, centers)
    num_iterations = 0
    while True:
        num_iterations += 1
        #want integral centers, regardless of whether dataset is naturally or artifically integral.
        centers = update_centers(dataset, codes, num_centers)#.round() #<--don't round, since quantizing after not before
        new_codes = compute_codes(dataset, centers)

        if torch.equal(codes, new_codes):
            #print('Converged in %d iterations' % num_iterations)
            break
        if num_iterations == MAX_KM_ITER:
            break
        codes = new_codes

    return centers, codes
    
