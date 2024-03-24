import argparse
import os

parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--device", type=str, default='cuda:0')
# parser.add_argument("--num_works", type=int, default=1)
# parser.add_argument('--checkpoints_dir', metavar='RESULTS_DIR',
#                     default='./checkpoints', help='checkpoints_dir')
# parser.add_argument('--logs_dir', default='./logs', help='logs_dir')
# parser.add_argument('--debug_mode', default=False, action='store_true')

# Data
parser.add_argument("--data_dir", type=str,
                    default="./data")
parser.add_argument("--num_points", type=int,
                    default=10)

# parser.add_argument('--img_size', type=int,
#                     default=224, help='input patch size of network input')
# parser.add_argument('--batch_size', type=int, default=5)
# parser.add_argument('--seed', type=int, default=1234)
# parser.add_argument("--enable_few_data", default=False, action='store_true')
# parser.add_argument("--fold", type=int)
# parser.add_argument('--sampling_k', type=int, default=10)
# parser.add_argument('--cross_vali_num', type=int, default=5)

# Model
# parser.add_argument("--model_name", type=str, default="UNet")
# parser.add_argument("--initial_filter_size", type=int, default=48)
parser.add_argument("--patch_size", nargs='+', default=, type=int)
parser.add_argument("--quantization_size", type=int, default=13)
parser.add_argument("--classes", type=int, default=13)
# parser.add_argument('--pick_y_methods', type=str)
# parser.add_argument('--pick_y_numbers', type=str)
# parser.add_argument('--ratios', type=str)
# parser.add_argument('--use_graph_flags', type=str)
# parser.add_argument('--num_layers', type=int, default=9,
#                     help='number of layers in the graph')


## TransUnet
# parser.add_argument('--n_skip', type=int,
#                     default=3, help='using number of skip-connect, default is num')
# parser.add_argument('--vit_name', type=str,
#                     default='R50-ViT-B_16', help='select one vit model')
# parser.add_argument('--vit_patches_size', type=int,
#                     default=16, help='vit_patches_size, default is 16')


# Train
# parser.add_argument("--experiment_name", type=str,
#                     default="")
# parser.add_argument("--restart", default=False, action='store_true')
# parser.add_argument("--pretrained_model_path", type=str,
#                     default='/afs/crc.nd.edu/user/d/dzeng2/UnsupervisedSegmentation/results/supervised_v3_train_2020-10-26_18-41-29/model/latest.pth')
# parser.add_argument("--epochs", type=int, default=100)
# parser.add_argument("--lr", type=float, default=1e-4)
# parser.add_argument("--min_lr", type=float, default=1e-6)
# parser.add_argument("--decay", type=str, default='50-100-150-200')
# parser.add_argument("--gamma", type=float, default=0.5)
# parser.add_argument("--optimizer", type=str, default='rmsprop',
#                     choices=('sgd', 'adam', 'rmsprop'))
# parser.add_argument("--weight_decay", type=float, default=1e-4)
# parser.add_argument("--momentum", type=float, default=0.9)
# parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
# parser.add_argument("--epsilon", type=float, default=1e-8)
# parser.add_argument("--do_contrast", default=False, action='store_true')
# parser.add_argument("--lr_scheduler", type=str, default='cos')
# parser.add_argument("--contrastive_method", type=str, default='simclr',
#                     help='simclr, gcl(global contrastive learning), pcl(positional contrastive learning)')
# parser.add_argument('--weight_cnn_contrast', type=float, default=1.0)
# parser.add_argument('--weight_graph_contrast', type=float, default=1.0)
# parser.add_argument('--weight_corr', type=float, default=1.0)
# parser.add_argument('--weight_local_contrast', type=float, default=1.0)
# parser.add_argument('--num_epoch_record', type=int, default=3)

# Loss
# parser.add_argument("--temp", type=float, default=0.1)
# parser.add_argument("--slice_threshold", type=float, default=0.05)


def save_args(obj, defaults, kwargs):
    for k, v in defaults.iteritems():
        if k in kwargs:
            v = kwargs[k]
        setattr(obj, k, v)


def get_config():
    config = parser.parse_args()
    config.data_dir = os.path.expanduser(config.data_dir)
    config.patch_size = tuple(config.patch_size)
    return config
