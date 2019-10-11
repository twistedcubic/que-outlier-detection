
'''
Processes data
'''
import torch
import numpy as np
import utils
import os.path as osp
import pdb

'''
Load subsampled genetics data.
'''
def clean_genetics_data():    
    data = np.loadtxt(osp.join(utils.data_dir, 'ALL.20k.data'), delimiter=' ')
    col_sums = np.sum((data==0).astype(np.int), axis=0)
    #remove any column containing 0, which indicates missing
    data = data[:, col_sums==0]
    #convert 2->0, 1->1
    data = 2 - data
    return data

def load_genetics_data():
    X = np.load(osp.join(utils.data_dir, 'sampled_data.npy'))
    X = torch.from_numpy(X).to(dtype=torch.float32, device=utils.device)
    return X

def load_glove_data():
    X = np.load('/home/yihdong/partition/data/glove_dataset.npy')
    X = torch.from_numpy(X).to(dtype=torch.float32, device=utils.device)
    return X

def process_glove_data(dim=100):
    path = osp.join(utils.data_dir, 'glove_embs.pt')
    if osp.exists(path):
        d = torch.load(path)
        #aa={'vocab':words_ar, 'word_emb':word_emb}
        return d['vocab'], d['word_emb'].to(utils.device)
    else:
        return load_process_glove_data(dim)
        
'''
Process glove vectors from raw txt file into numpy arrays.
'''
def load_process_glove_data(dim=100):
    path = osp.join(utils.data_dir, 'glove.6B.{}d.txt'.format(dim))
    lines = load_lines(path)
    lines_len = len(lines)
    words_ar = [0]*lines_len
    word_emb = torch.zeros(lines_len, dim)
    for i, line in enumerate(lines):
        line_ar = line.split()
        words_ar[i] = line_ar[0]
        word_emb[i] = torch.FloatTensor([float(t) for t in line_ar[1:]])
    
    word_emb = word_emb.to(utils.device)
    return words_ar, word_emb

def load_lines(path):
    with open(path, 'r') as file:
        lines = file.read().splitlines()
    return lines

def write_lines(lines1, path):
    lines = []
    for line in lines1:
        lines.append(str(line) + os.linesep)   
    with open(path, 'w') as file:
        file.writelines(lines)
    
if __name__ == '__main__':
    data = load_genetics_data()
    pdb.set_trace()
