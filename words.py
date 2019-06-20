
'''
Detect outliers in word embeddings
'''
import torch
import allennlp.data.tokenizers.word_tokenizer as tokenizer
from allennlp.data.tokenizers.word_filter import StopwordFilter
import sklearn.decomposition as decom
import data
import utils
import numpy as np
import numpy.linalg as linalg
import re

import pdb

'''
Combines content and noise words embeddings
'''
def doc_word_embed_content_noise(content_path, noise_path, whiten_path=None, content_lines=None, noise_lines=None, opt=None):
    no_add_set = set()
    doc_word_embed_f = doc_word_embed_sen
    content_words_ar, content_word_embeds = doc_word_embed_f(content_path, no_add_set, content_lines=content_lines)
    words_set = set(content_words_ar)
    noise_words_ar, noise_word_embeds = doc_word_embed_f(noise_path, set(content_words_ar), content_lines=noise_lines)
    content_words_ar.extend(noise_words_ar)
    words_ar = content_words_ar
    word_embeds = torch.cat((content_word_embeds, noise_word_embeds), dim=0)
    
    whitening = opt.whiten if opt is not None else True  #True #April, temporary normalize by inlier covariance!
    if whitening and whiten_path is not None:
        #use an article of data in the inliers topic to whiten data.
        whiten_ar, whiten_word_embeds = doc_word_embed_f(whiten_path, set()) #, content_lines=content_lines)#,content_lines=content_lines) ######april!!
        
        whiten_cov = utils.cov(whiten_word_embeds)
        fast_whiten = False #True
        if not fast_whiten:
            U, D, V_t = linalg.svd(whiten_cov)
            #D_avg = D.mean() #D[len(D)//2]
            #print('D_avg! {}'.format(D_avg))
            
            cov_inv = torch.from_numpy(np.matmul(linalg.pinv(np.diag(np.sqrt(D))), U.transpose())).to(utils.device)
            #cov_inv = torch.from_numpy(np.matmul(U, np.matmul(linalg.pinv(np.diag(np.sqrt(D))), V_t))).to(utils.device)

            word_embeds0=word_embeds
            #change multiplication order!
            word_embeds = torch.mm(cov_inv, word_embeds.t()).t()
            if False:
                
                after_cov = utils.cov(word_embeds)
                U1, D1, V_t1 = linalg.svd(after_cov)                
                pdb.set_trace()
                
                content_whitened = torch.mm(cov_inv, content_word_embeds.t()).t()
                after_cov2 = utils.cov(content_whitened)
                _, D1, _ = linalg.svd(after_cov2) 
                print('after whitening D {}'.format(D1[:7]))
        else:
            #### faster whitening
            sv = decom.TruncatedSVD(30)
            sv.fit(whiten_cov.cpu().numpy())
            top_evals, top_evecs = sv.singular_values_, sv.components_
            top_evals = torch.from_numpy(1/np.sqrt(top_evals)).to(utils.device)
            top_evecs = torch.from_numpy(top_evecs).to(utils.device)
            #pdb.set_trace()
            
            X = word_embeds
            projected = torch.mm(top_evecs.t()/(top_evecs**2).sum(-1), torch.mm(top_evecs, X.t())).t()
            #eval_ones = torch.eye(len(top_evals), device=top_evals.device)
            ##projected = torch.mm(torch.mm(top_evecs.t(), eval_ones), torch.mm(top_evecs, X.t())).t()
            
            #(d x k) * (k x d) * (d x n), project onto and squeeze the components along top evecs
            ##word_embeds = torch.mm((top_evecs/top_evals.unsqueeze(-1)).t(), torch.mm(top_evecs, X.t())).t() + (X-torch.mm(top_evecs.t(), torch.mm(top_evecs, X.t()) ).t())
            #pdb.set_trace()
            ##word_embeds = torch.mm((top_evecs/(top_evals*(top_evecs**2).sum(-1)).unsqueeze(-1)).t(), torch.mm(top_evecs, X.t())).t() + (X-projected )
            #word_embeds = torch.mm((top_evecs/(top_evals*(top_evecs**2).sum(-1)).unsqueeze(-1)).t(), torch.mm(top_evecs, X.t())).t() + (X-projected )
            word_embeds = torch.mm(torch.mm(top_evecs.t(), top_evals.diag()), torch.mm(top_evecs, X.t())).t() + (X-projected )            
    
    noise_idx = torch.LongTensor(list(range(len(content_word_embeds), len(word_embeds)))).to(utils.device)
    if False:
        #normalie per direction
        word_embeds_norm = ((word_embeds-word_embeds.mean(0))**2).sum(dim=1, keepdim=True).sqrt()
    debug_top_dir = False
    if debug_top_dir:
        w1 = (content_word_embeds - word_embeds.mean(0))#/word_embeds_norm[:len(content_word_embeds)]
        
        w2 = (noise_word_embeds - word_embeds.mean(0))#/word_embeds_norm[len(content_word_embeds):]
        mean_diff = ((w1.mean(0)-w2.mean(0))**2).sum().sqrt()
        w1_norm = (w1**2).sum(-1).sqrt().mean()
        w2_norm = (w2**2).sum(-1).sqrt().mean()
        X = (word_embeds - word_embeds.mean(0))#/word_embeds_norm
        cov = torch.mm(X.t(), X)/word_embeds.size(0)
        U, D, V_t = linalg.svd(cov.cpu().numpy())
        U1 = torch.from_numpy(U[1]).to(utils.device)
        mean1_dir = w1.mean(0)
        mean1_proj = (mean1_dir*U1).sum()
        mean2_dir = w2.mean(0)
        mean2_proj = (mean2_dir*U1).sum()
        diff_proj = ((mean1_dir-mean2_dir)*U1).sum()

        #plot histogram of these projections
        proj1 = (w1*U1).sum(-1)
        proj2 = (w2*U1).sum(-1)
        utils.hist(proj1, 'inliers')
        utils.hist(proj2, 'outliers')
        pdb.set_trace()
    #word_embeds=(word_embeds - word_embeds.mean(0))/word_embeds_norm
    return words_ar, word_embeds, noise_idx

'''
Read in file, get embeddings, remove stop words
Input:
-no_add_set: words to not add
'''
def doc_word_embed(path, no_add_set, content_lines=None):
    if content_lines is not None:
        lines = content_lines
    else:
        with open(path, 'r') as file:
            lines = file.readlines()

    lines1 = []
    words = []
    vocab, embeds = data.process_glove_data(dim=100)
    embed_map = dict(zip(vocab, embeds))
    
    tk = tokenizer.WordTokenizer()
    #list of list of tokens
    tokens_l = tk.batch_tokenize(lines)
    stop_word_filter = StopwordFilter()

    tokens_l1 = []
    for sentence_l in tokens_l:
        tokens_l1.extend(sentence_l)
    tokens_l = [tokens_l1]

    n_avg = 5 #5
    word_embeds = []
    words_ar = []
    added_set = set(no_add_set)
    for sentence in tokens_l:
        
        sentence = stop_word_filter.filter_words(sentence)
        cur_embed = torch.zeros_like(embed_map['a'])
        cur_counter = 0
        for j,w in enumerate(sentence):
            w = w.text.lower()
            if w in embed_map:# and w not in added_set:
                if cur_counter == n_avg:# or j==len(sentence)-1:
                    added_set.add(w)
                    words_ar.append(w)
                    #word_embeds.append(embed_map[w])
                    #word_embeds.append(cur_embed/(cur_counter if cur_counter > 0 else 1))
                    word_embeds.append(cur_embed/n_avg)
                    
                    cur_embed = torch.zeros_like(embed_map['a'])
                    cur_counter = 0      
                else:
                    cur_counter += 1
                    cur_embed += embed_map[w]
    
    word_embeds = torch.stack(word_embeds, dim=0).to(utils.device)    
    if False: #is_noise :#False: #sanity check
        word_embeds[:] = word_embeds.mean(0) #word_embeds[0]
    return words_ar, word_embeds
'''
embedding of sentences.
'''
def doc_word_embed_sen(path, no_add_set, content_lines=None):
    if content_lines is not None:
        lines = content_lines
    else:
        with open(path, 'r') as file:
            lines = file.readlines()
            
    lines1 = []
    patt = re.compile('[;\.:!,?]')
    for line in lines:
        #cur_lines = []
        for cur_line in patt.split(line):
            lines1.append(cur_line)            
        #lines1.append(cur_lines)
    lines = lines1    
    
    words = []
    vocab, embeds = data.process_glove_data(dim=100)
    embed_map = dict(zip(vocab, embeds))
    
    tk = tokenizer.WordTokenizer()
    #list of list of tokens
    tokens_l = tk.batch_tokenize(lines)
    stop_word_filter = StopwordFilter()

    '''
    tokens_l1 = []
    for sentence_l in tokens_l:
        tokens_l1.extend(sentence_l)
    tokens_l = [tokens_l1]
    '''
    max_len = 200
    word_embeds = []
    words_ar = []
    #added_set = set(no_add_set)
    for sentence in tokens_l:        
        sentence = stop_word_filter.filter_words(sentence)        
        if len(sentence) < 4:
            continue
        cur_embed = torch.zeros_like(embed_map['a'])
        cur_counter = 0
        for j,w in enumerate(sentence):
            w = w.text.lower()
            if w in embed_map:# and w not in added_set:
                if cur_counter == max_len:# or j==len(sentence)-1:
                    #added_set.add(w)
                    words_ar.append(w)
                    #word_embeds.append(embed_map[w])
                    #word_embeds.append(cur_embed/(cur_counter if cur_counter > 0 else 1))
                    word_embeds.append(cur_embed/max_len)
                    
                    cur_embed = torch.zeros_like(embed_map['a'])
                    cur_counter = 0      
                else:
                    cur_counter += 1
                    cur_embed += embed_map[w]
        
        word_embeds.append(cur_embed / len(sentence))
        
    word_embeds = torch.stack(word_embeds, dim=0).to(utils.device)    
    if False: #is_noise :#False: #sanity check
        word_embeds[:] = word_embeds.mean(0) #word_embeds[0]
    
    return words_ar, word_embeds

def doc_word_embed0(path, no_add_set):
    with open(path, 'r') as file:
        lines = file.readlines()

    lines1 = []
    #for line in lines:
    #    lines1.extend(line.lower().split('.') )        
    #lines = lines1
    
    words = []
    vocab, embeds = data.process_glove_data(dim=100)
    embed_map = dict(zip(vocab, embeds))
    
    tk = tokenizer.WordTokenizer()
    #list of list of tokens
    tokens_l = tk.batch_tokenize(lines)
    stop_word_filter = StopwordFilter()
    
    word_embeds = []
    words_ar = []
    added_set = set(no_add_set)
    for sentence in tokens_l:
        sentence = stop_word_filter.filter_words(sentence)    
        for w in sentence:
            w = w.text.lower()
            if w in embed_map and w not in added_set:
                added_set.add(w)
                words_ar.append(w)
                word_embeds.append(embed_map[w])
      
    word_embeds = torch.stack(word_embeds, dim=0).to(utils.device)
    if False: #sanity check
        word_embeds[:] = word_embeds[0] 
    #word_embeds = word_embeds / (word_embeds**2).sum(dim=1, keepdim=True).sqrt()
    return words_ar, word_embeds

'''
Create sentence embeddings on the file with the supplied path.
'''
def doc_sentence_embed(path):
    with open(path, 'r') as file:
        lines = file.readlines()

    lines1 = []
    for line in lines:
        lines1.extend(line.lower().split('.') )
        
    lines = lines1
    words = []
    vocab, embeds = data.process_glove_data(dim=100)
    embed_map = dict(zip(vocab, embeds))
    
    tk = tokenizer.WordTokenizer()
    tokens_l = tk.batch_tokenize(lines)
    word_embeds = []
    words_ar = []
    added_set = set()
    for sentence in tokens_l:
        if len(sentence) < 3:
            continue
        sentence_embed = 0
        aa = True
        for w in sentence:
            w = w.text.lower()
            if w in embed_map:# and w not in added_set:
                ##added_set.add(w)
                ##words_ar.append(w)
                ##word_embeds.append(embed_map[w])
                sentence_embed += embed_map[w]
                aa = False
        if aa:
            continue
        words_ar.append(sentence)
        word_embeds.append(sentence_embed/len(sentence))

    word_embeds = torch.stack(word_embeds, dim=0).to(utils.device)
    #word_embeds = word_embeds / (word_embeds**2).sum(dim=1, keepdim=True).sqrt()
    return words_ar, word_embeds

if __name__=='__main__':
    doc_word_embed('data/test.txt', set())
    
