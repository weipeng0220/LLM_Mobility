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
        parser.add_argument('--path_traj', type=str, default='/data3/weipeng/workspace/LLM_Mobility/data/sh/CDRsh_traj.pkl', help='path of trajectory')
        parser.add_argument('--path_attr', type=str, default='/data3/weipeng/workspace/LLM_Mobility/data/sh/CDRsh_attr.pkl', help='path of attr')
        parser.add_argument('--uid_mask_day', type=str, default='/data3/weipeng/workspace/LLM_Mobility/data/sh/CDRsh_mask_day.pkl', help='path of uid_mask_day')
        parser.add_argument('--path_loccoor', type=str, default='/data3/weipeng/workspace/LLM_Mobility/data/sh/CDRsh_loccoor.pkl', help='path of loc coordinate')
        parser.add_argument('--loc_max', type=int, default=16050, help='size of location id')
        parser.add_argument('--user_max', type=int, default=20000, help='size of user id')
        parser.add_argument('--traj_length', type=int, default=24, help='the length of trajectory')
        parser.add_argument('--padding_value', type=int, default=0, help='padding value')
    ## model
    parser.add_argument('--z_latent_size', type=int, default=512, help='size of latent z')
    parser.add_argument('--z_hidden_size', type=int, default=512, help='size of hidden z')
    parser.add_argument('--hidden_size', type=int, default=512, help='size of hidden')
    parser.add_argument('--embedding_dim', type=int, default=512, help='size of embedding loc')
    parser.add_argument('--dim_forward', type=int, default=512, help='size of transformer encoder forward dim')
    parser.add_argument('--head_num', type=int, default=4, help='transformer encoder head number')
    parser.add_argument('--drop_out', type=float, default=0.1, help='drop out')
    parser.add_argument('--layer_num', type=int, default=3, help='number of transformer layers')
    parser.add_argument('--clamp_min', type=float, default=1e-4, help='clamp min')
    ## train
    parser.add_argument('--gpu', type=str, default=0, help='GPU index to choose')
    parser.add_argument('--epoch_num', type=int, default=3, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=10, help='train batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Type of optimizer, choice is adam, adagrad, rmsprop')
    parser.add_argument('--path_out', type=str, default=f'../results/{city}/', help='output data path')
    parser.add_argument('--path_save', type=str, default=f'../save/{city}/', help='model saved path')
    ## lora
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=list, default=["q_proj", "k_proj", "v_proj", "o_proj"], help='target modules')
    parser.add_argument('--lora_bias', type=str, default="none", help='lora_bias')
    parser.add_argument('--task_type', type=str, default="CAUSAL_LM", help='lora task type')


    if params == None:
        params = parser.parse_args()
    else:
        params = parser.parse_args(params)

    if params.gpu == -1:
        params.device = torch.device("cpu")
    else:
        params.device = torch.device("cuda", int(params.gpu))

    print('Data', time.strftime("%Y%m%d"))

    params.path_out = f'{params.path_out}{time.strftime("%Y%m%d")}/'
    if not os.path.exists(params.path_out):
        os.makedirs(params.path_out)

    params.path_save = f'{params.path_save}{time.strftime("%Y%m%d")}/'
    if not os.path.exists(params.path_save):
        os.makedirs(params.path_save)

    print('params', params)
    extra_args, remaining_args = parser.parse_known_args()

    return params, remaining_args
