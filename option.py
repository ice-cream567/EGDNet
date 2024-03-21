import argparse

parser = argparse.ArgumentParser(description='EG_Net')
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--batch_size', type=int, default=1)#128
parser.add_argument('--test_only',default=True, help='set this option to test the model')#,action='store_true'
# parser.add_argument('--iters_per_epoch', type=int, default=10000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=0)

#Data specifications
parser.add_argument('--data_dir', type=str, default='./data/datasets')
parser.add_argument('--save_dir', type=str, default='./saved_models/esl')
parser.add_argument('--load_dir', type=str, default='./saved_models/esl/49')
# Model specifications

args = parser.parse_args()
