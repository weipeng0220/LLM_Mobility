import pickle
import numpy as np
import torch
from tqdm import tqdm
import os
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def get_batch_home_info(uid_traj,uid_mask_day,user_attr,split = [0.6, 0.1, 0.3]):
    batch_all = {'train': [], 'valid': [], 'test': []}
    for uid, traj_all in tqdm(list(uid_traj.items())[:]):
        traj_num = len(traj_all)
        valid_num = traj_num
        num_train = int(valid_num * split[0])
        num_valid = int(valid_num * split[1])
        num_test = valid_num - num_train - num_valid
        for idx in range(num_train):
            if uid_mask_day[uid][idx] == 0:
                batch_all['train'].append((uid,idx,user_attr[uid]['home']))
        for idx in range(num_valid):
            if uid_mask_day[uid][idx] == 0:
                batch_all['valid'].append((uid,idx,user_attr[uid]['home']))
        for idx in range(num_test):
            if uid_mask_day[uid][idx] == 0:
                batch_all['test'].append((uid,idx,user_attr[uid]['home']))
    return batch_all

def collate_batch_data(uid_traj,batch_all,params):
    random.shuffle(batch_all)
    batch_num = round(len(batch_all) / params.batch_size)
    print('Batch number is ', batch_num)

    for i in range(batch_num):
        # if i == 0:
        #     print('Batch:', end=' ')
        # elif batch_num < 100:
        #     print(i, end=',')
        # elif i > 0 and i % (batch_num // 100) == 0:
        #     print(f'{i / batch_num * 100:.0f}%', end=',')
        # if i == batch_num - 1:
        #     print('Batch End')

        batch_start, batch_end = i * params.batch_size, (i + 1) * params.batch_size
        batch_list = batch_all[batch_start:batch_end]

        trajs=[]
        home_ids=[]

        for uid,idx,home in batch_list:
            assert len(uid_traj[uid][idx])==params.traj_length
            trajs.append(torch.LongTensor(uid_traj[uid][idx]))
            home_ids.append(home)

        trajs_tensor=torch.stack(trajs, dim=0)
        home_ids_tensor=torch.LongTensor(home_ids)

        yield trajs_tensor, home_ids_tensor







