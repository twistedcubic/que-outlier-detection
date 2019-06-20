import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import scipy as sp
import sklearn.decomposition as decom
import pdb

'''
Must have CIFAR images downloaded. E.g. from https://www.cs.toronto.edu/~kriz/cifar.html.
'''

D = 1024
PIXEL_VALUE_RANGE  = 256
VARIANCE_OUTLIER_DISTRIBUTION = 20

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# returns a pair (cifar,airplanes) of all cifar data and airplanes
def init():
    cifar1 = unpickle('data/cifar-10-batches-py/data_batch_1')
    cifar2 = unpickle('data/cifar-10-batches-py/data_batch_2')
    cifar3 = unpickle('data/cifar-10-batches-py/data_batch_3')
    cifar4 = unpickle('data/cifar-10-batches-py/data_batch_4')
    cifar5 = unpickle('data/cifar-10-batches-py/data_batch_5')
    cifar6 = unpickle('data/cifar-10-batches-py/test_batch')

    cifar = np.concatenate((cifar1[b'data'], cifar2[b'data'], cifar3[b'data'], cifar4[b'data'], cifar5[b'data'], cifar6[b'data']))
    cifar_red = cifar[:,0:1024].astype(int) #keep only the red channel to speed things up

    # get only the airplanes
    cifar_by_label = [[] for i in range(10)]
    for batch in [cifar1,cifar2,cifar3,cifar4,cifar5,cifar6]:
        for i in range(len(batch[b'labels'])):
            cifar_by_label[batch[b'labels'][i]].append(batch[b'data'][i])

    for i in range(len(cifar_by_label)):
        cifar_by_label[i] = np.array([y[:1024].astype(int) for y in cifar_by_label[i]])


    # sort into classes with some incredibly garbage code
    class0 = []
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    class5 = []
    class6 = []
    class7 = []
    class8 = []
    class9 = []

    for batch in [cifar1,cifar2,cifar3,cifar4,cifar5,cifar6]:
        for i in range(len(batch[b'labels'])):
            if batch[b'labels'][i] == 0:
                class0 += [batch[b'data'][i][:1024].astype(int)]
            if batch[b'labels'][i] == 1:
                class1 += [batch[b'data'][i][:1024].astype(int)]
            if batch[b'labels'][i] == 2:
                class2 += [batch[b'data'][i][:1024].astype(int)]
            if batch[b'labels'][i] == 3:
                class3 += [batch[b'data'][i][:1024].astype(int)]
            if batch[b'labels'][i] == 4:
                class4 += [batch[b'data'][i][:1024].astype(int)]
            if batch[b'labels'][i] == 5:
                class5 += [batch[b'data'][i][:1024].astype(int)]
            if batch[b'labels'][i] == 6:
                class6 += [batch[b'data'][i][:1024].astype(int)]
            if batch[b'labels'][i] == 7:
                class7 += [batch[b'data'][i][:1024].astype(int)]
            if batch[b'labels'][i] == 8:
                class8 += [batch[b'data'][i][:1024].astype(int)]
            if batch[b'labels'][i] == 9:
                class9 += [batch[b'data'][i][:1024].astype(int)]

    cifar_by_class = [np.array(cl) for cl in [class0,class1,class2,class3,class4,class5,class6,class7,class8,class9]]

    return cifar_red,cifar_by_class


# randomly subsamples n elements of np array, returns np array
def subsample(array,n,sample_idx=None):
    if sample_idx is not None:
        return array[sample_idx]
    else:
        return np.array([array[np.random.randint(array.shape[0])] for i in range(n)])


'''
Computes data used for whitening.
Input:
-data: image data
-fast_whiten: whether to do approximate or exact whitening.
-sample_idx: indices used for whitening
Returns: 
-whitening matrix, computed with either exact or appx inverse and 5000 random cifar images from all classes
'''
def get_whitening(data, fast_whiten=False, sample_idx=None):

    N = 5000

    cifar_red,airplanes = data

    # subsample
    whitening_imgs = subsample(cifar_red,N,sample_idx)

    w_mean = np.sum(whitening_imgs,axis=0) / whitening_imgs.shape[0]
    w_centered = whitening_imgs - np.outer(np.ones(whitening_imgs.shape[0]), w_mean)
    w_cov = np.dot(np.transpose(w_centered), w_centered) / whitening_imgs.shape[0]
    
    if fast_whiten:
        whiten_dim = int(0.3*cifar_red.shape[-1])
        sv = decom.TruncatedSVD(whiten_dim)
        sv.fit(w_cov)
        
        top_evals, top_evecs = sv.singular_values_, sv.components_
        top_evals = 1/np.sqrt(top_evals)
        
        return (top_evals, top_evecs)
        
    else:
        whiten = sp.linalg.sqrtm(w_cov)
        whiten = np.linalg.inv(whiten)
        return whiten


def get_corrupted_data(data,num_directions,frac_bad,W,one_class=False,which_class=1,fast_whiten=False,sample_idx=None):
    '''
    args:
      data -- a pair cifar,airplanes consisting of all cifar images and just the airplane images
      num_directions -- how many different bad pixels to make
      frac_bad -- what percent of data should be outliers
      W -- a whitening matrix
      one_class -- use only one class of images. IN THIS CASE WE DO NOT RANDOMLY SUBSAMPLE

    returns:
      good_data,bad_data -- a pair of numpy arrays with good whitened data and bad whitened data

    we randomly subsample 5000 cifar images and corrupt a subset of them

    '''

    imgs,by_label= data

     # make a fresh copy of the data so we can modify it
    imgs = np.copy(imgs)

    # randomly subsample 5000 images
    imgs = subsample(imgs,5000,sample_idx)

    # if using one class of images
    if one_class:
        imgs = np.copy(by_label[which_class])

    # split the data
    num_bad = int(frac_bad * len(imgs))
    num_good = len(imgs) - num_bad

    bad_data = imgs[:num_bad]
    good_data = imgs[num_bad:]

    # compute how many of each outlier type
    #hot_fracs = [np.random.randint(VARIANCE_OUTLIER_DISTRIBUTION) for i in range(num_directions)]
    hot_fracs = [1.3**i for i in range(num_directions)]
    s = float(sum(hot_fracs))
    hot_fracs = map(lambda x: x/s, hot_fracs)
    hot_nums = [int(x * num_bad) for x in hot_fracs]

    # introduce corruptions
    for i in range(num_directions):

        # pick locations and values for hot pixels at random
        px_val = np.random.randint(PIXEL_VALUE_RANGE)
        px_loc = np.random.randint(D)
        start_idx = sum(hot_nums[:i])
        
        #for j in range(start_idx, start_idx+ ):
        for j in range(start_idx, start_idx+hot_nums[i]):
            bad_data[j][px_loc] = px_val
        #start_idx = end_idx

    # whiten and center the data
    if fast_whiten:
        top_evals, top_evecs = W
        X = np.concatenate((bad_data,good_data),axis=0)
        projected = np.matmul(top_evecs.transpose()/(top_evecs**2).sum(-1), np.matmul(top_evecs, X.transpose())).transpose()
        
        X = np.matmul(np.matmul(top_evecs.transpose(), np.diag(top_evals)), np.matmul(top_evecs, X.transpose())).transpose() + (X-projected )
        bad_data_w = X[:len(bad_data)]
        good_data_w = X[len(bad_data):]
    else:
        bad_data_w = np.dot(bad_data, W)
        good_data_w= np.dot(good_data, W)

    mean = np.sum(np.concatenate((bad_data_w,good_data_w),axis=0), axis=0) / imgs.shape[0]
    bad_data_cw = bad_data_w - np.outer(np.ones(bad_data_w.shape[0]), mean)
    good_data_cw = good_data_w - np.outer(np.ones(good_data_w.shape[0]), mean)

    return good_data_cw,bad_data_cw
