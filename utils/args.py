import argparse
import os
import torch
import time

def param_settings(city, params=None):
    parser = argparse.ArgumentParser()
    if city == 'SH':
        ## data
        parser.add_argument('--city', type=str, default=city, help='dataset name')
        parser.add_argument('--data_path', type=str, default='data/', help='path of trajectory')
        parser.add_argument('--path_traj', type=str, default='data/SH_traj.pkl', help='path of trajectory')
        parser.add_argument('--path_traj_split', type=str,
                            default='data/SH_traj_rl.pkl',
                            help='path of trajectory rl')
        parser.add_argument('--path_attr', type=str, default='data/SH_attr.pkl', help='path of attr')
        parser.add_argument('--uid_mask_day', type=str, default='data/SH_mask_day.pkl', help='path of uid_mask_day')
        parser.add_argument('--path_loccoor', type=str, default='data/SH_loccoor.pkl', help='path of loc coordinate')
        parser.add_argument('--loc_max', type=int, default=16050, help='size of location id')
        parser.add_argument('--user_max', type=int, default=20000, help='size of user id')
        parser.add_argument('--dis_matrix', type=str, default='data/dis_matrix.pkl', help='path of distance matrix')
    ## model
    parser.add_argument('--z_latent_size', type=int, default=512, help='size of latent z')
    parser.add_argument('--z_hidden_size', type=int, default=512, help='size of hidden z')
    parser.add_argument('--hidden_size', type=int, default=512, help='size of hidden')
    parser.add_argument('--embedding_size', type=int, default=512, help='size of embedding loc')
    parser.add_argument('--dim_forward', type=int, default=512, help='size of transformer encoder forward dim')
    parser.add_argument('--head_num', type=int, default=4, help='transformer encoder head number')
    parser.add_argument('--drop_out', type=float, default=0.1, help='drop out')
    parser.add_argument('--layer_num', type=int, default=3, help='number of transformer layers')
    ## train
    parser.add_argument('--gpu', type=str, default=1, help='GPU index to choose')
    parser.add_argument('--epoch_num', type=int, default=30, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=512, help='train batch size')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Type of optimizer, choice is adam, adagrad, rmsprop')
    parser.add_argument('--path_out', type=str, default=f'../results/{city}/', help='output data path')
    ## lora
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=int, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=list, default=["q_proj", "k_proj", "v_proj", "o_proj"], help='target modules')
    parser.add_argument('--lora_bias', type=str, default="none", help='lora_bias')
    parser.add_argument('--task_type', type=str, default="CAUSAL_LM", help='lora task type')


    if params.gpu == -1:
        params.device = torch.device("cpu")
    else:
        params.device = torch.device("cuda", int(params.gpu))

    if params == None:
        params = parser.parse_args()
    else:
        params = parser.parse_args(params)

    params.path_out = f'{params.path_out}{time.strftime("%Y%m%d")}/'

    print('Data', time.strftime("%Y%m%d"))

    if not os.path.exists(params.path_out):
        os.makedirs(params.path_out)

    print('params', params)

    return params
