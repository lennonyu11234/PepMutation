import argparse
import warnings
import os
import torch
import sys
sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='regression', help='')
parser.add_argument('--threshold', type=int, default=16, help='')
parser.add_argument('--max_len', type=int, default=50, help='')
parser.add_argument('--vocab_size', type=int, default=24, help='')
parser.add_argument('--emb_dim', type=int, default=512, help='')
parser.add_argument('--batch_size', type=int, default=512, help='')
parser.add_argument('--lr', type=float, default=1e-4, help='')
parser.add_argument('--epochs', type=int, default=1000, help='')
parser.add_argument('--model_save_dir', type=str, default='model/RNN-Prior.pth', help='')
parser.add_argument('--data_path', type=str, default='Data/Dataset/Pretrain.csv', help='')
parser.add_argument('--load_model', type=bool, default=True, help='')

parser.add_argument('--num_heads', type=int, default=8, help='')
parser.add_argument('--num_layers', type=int, default=4, help='')

parser.add_argument('--train_ratio', type=float, default=0.8, help='')
parser.add_argument('--val_ratio', type=float, default=0.1, help='')
parser.add_argument('--test_ratio', type=float, default=0.1, help='')

parser.add_argument('--multi_func_names', type=str, default='E_coli', help='')
parser.add_argument('--multi_weights', type=str, default='0.8', help='')
parser.add_argument('--policy_save_path', type=str, default='Data/model/policy_net.pth')
parser.add_argument('--change_rate_path', type=str, default='Data/result/mutation_change_rate/change_rate.csv')
parser.add_argument('--sample_path', type=str, default='result/sample.csv')
parser.add_argument('--tokenizer_path', type=str, default='Pep')



args = parser.parse_args()
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
