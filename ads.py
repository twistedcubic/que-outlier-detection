
'''
Process ads data
'''
import torch
import utils

import pdb

'''
file in arff format
'''
def get_data(path):
    with open(path) as file:
        lines = file.readlines()
        
    data_l = []
    #bool_l = []
    noise_idx_l = []
    id_l = []
    counter = 0
    for line in lines:
        if line[0] == '%' or line[0] == '@' or len(line)<5:
            continue
        line_ar = line.split(',')

        #second to last is id, some integer like 175, "@ATTRIBUTE 'id' real\n"
        data_l.append([float(i) for i in line_ar[:-2]])
        id_l.append(float(line_ar[-2]))
        if line_ar[-1] == "'yes'\n":
            noise_idx_l.append(counter)
            #bool_l.append(1)
        #else:
        #    bool_l.append(0)
        counter += 1
        
    data = torch.FloatTensor(data_l).to(utils.device)
    #is_ad = torch.IntTensor(bool_l).to(utils.device)
    noise_idx = torch.LongTensor(noise_idx_l).to(utils.device)
    
    return data, noise_idx

if __name__ == '__main__':
    data, noise_idx = get_ads('data/internet_ads.arff')
    pdb.set_trace()
