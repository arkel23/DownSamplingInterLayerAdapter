import os
import argparse
import torch

from .yaml_config_hook import yaml_config_hook


VITS = ['vit_t4', 'vit_t8', 'vit_t16', 'vit_t32', 'vit_s8', 'vit_s16', 'vit_s32',
        'vit_b8', 'vit_b16', 'vit_b32', 'vit_l16', 'vit_l32', 'vit_h14']
REDUCERS = (None, False, True, 'pool', 'cls')
HEADS = (None, 'cls', 'pool', 'blp', 'lrblp', 'iblp', 'mpncov', 'flatten', 'avg_cls', 'max_cls',
         'ifacls', 'ifaclsv2', 'ifaclsv3', 'ifaclsv4', 'ifaclsconv', 'ifaclsflatten')
AUX = (None, 'cls', 'pool', 'cont', 'multi_cont', 'shared', 'consistency')
SIM_METRICS = ('cos', 'l1', 'l2')
VIS_LIST = ('glsim_norm', 'rollout', 'psm_0', 'psm_11', 
            'maws_0', 'maws_11', 'attention_0', 'attention_11')


def add_adjust_common_dependent(args):
    if args.vis_errors_save:
        args.vis_errors = True

    if args.vis_errors:
        args.test_only_bs1 = True

    if args.test_only_bs1:
        print('When using test only changes batch size to 1 to simulate streaming')
        args.test_only = True
        args.batch_size = 1

    args.effective_batch_size = args.batch_size * args.gradient_accumulation_steps

    if args.base_lr:
        args.lr = args.base_lr * (args.effective_batch_size / 8)

    # distributed
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        args.device = torch.device(f'cuda:{args.local_rank}')
        torch.cuda.set_device(args.device)

        args.effective_batch_size = args.effective_batch_size * args.world_size

        if args.base_lr:
            args.lr = args.base_lr * (args.effective_batch_size / 8)

    if not args.resize_size:
        if args.image_size == 32:
            args.resize_size = 36
        elif args.image_size == 64:
            args.resize_size = 72
        elif args.dataset_name == 'ncfm':
            args.resize_size = args.image_size + 32
        else:
            args.resize_size = int(args.image_size * 1.34)

    if not args.test_resize_size:
        args.test_resize_size = args.resize_size

    if args.dataset_name and ('vtab' in args.dataset_name):
        args.train_resize_directly = True
        args.warmup_steps = 0
        args.warmup_epochs = 10
        args.horizontal_flip = False

    if not (args.train_resize_directly or args.random_resized_crop or args.square_resize_random_crop or
            args.short_side_resize_random_crop or args.square_center_crop):
        print('Needs at least one crop, using square_resize_random_crop by default')
        args.square_resize_random_crop = True

    if args.dataset_name == 'dogs':
        args.selector_num_tokens *= 2
    if args.image_size >= 448 and args.selector in ('rollout', 'glsim'):
        args.selector_num_tokens *= 2

    assert not (args.distributed and args.test_only), 'test_only cannot be used with multi gpu'

    return args


def add_common_args():
    parser = argparse.ArgumentParser('Arguments for code: FGIR-ViT')
    # general
    parser.add_argument('--project_name', type=str, default='ILA',
                        help='project name for wandb')
    parser.add_argument('--entity', type=str, default='nycu_pcs', help='wandb entity')
    parser.add_argument('--debugging', action='store_true',
                        help='when true disables wandb and exits after a single pass')
    parser.add_argument('--serial', default=0, type=int, help='serial number for run')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--log_freq', type=int,
                        default=100, help='print frequency (iters)')
    parser.add_argument('--save_freq', type=int,
                        default=200, help='save frequency (epochs)')
    parser.add_argument('--results_dir', type=str, default='results_train',
                        help='dir to save models from base_path')
    parser.add_argument('--per_class_acc_results', type=str, default='per_class_acc.csv')
    parser.add_argument('--ind_preds_results', type=str, default='ind_preds.csv')
    return parser


def add_data_args(parser):
    parser.add_argument('--image_size', type=int, default=224, help='image_size')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--cpu_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--custom_mean_std', action='store_true', help='custom mean/std')
    parser.add_argument('--pin_memory', action='store_false', help='pin memory for gpu (def: true)')
    parser.add_argument('--ffcv', action='store_true', help='speed up training')
    # dataset
    parser.add_argument('--shuffle_test', action='store_true',
                        help='if true then shuffles test data')
    parser.add_argument('--dataset_name', default=None, type=str, help='dataset name')
    parser.add_argument('--dataset_root_path', type=str, default=None,
                        help='the root directory for where the data/feature/label files are')
    # folders with images (can be same: those where it's all stored in 'data')
    parser.add_argument('--folder_train', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/train/')
    parser.add_argument('--folder_val', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/val/')
    parser.add_argument('--folder_test', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/test/')
    # df files with img_dir, class_id
    parser.add_argument('--df_train', type=str, default='train.csv',
                        help='the df csv with img_dirs, targets, def: train.csv')
    parser.add_argument('--df_trainval', type=str, default='train_val.csv',
                        help='the df csv with img_dirs, targets, def: train_val.csv')
    parser.add_argument('--df_val', type=str, default='val.csv',
                        help='the df csv with img_dirs, targets, def: val.csv')
    parser.add_argument('--df_test', type=str, default='test.csv',
                        help='the df csv with img_dirs, targets, root/test.csv')
    parser.add_argument('--df_classid_classname', type=str, default='classid_classname.csv',
                        help='the df csv with classnames and class ids, root/classid_classname.csv')
    return parser


def add_optim_scheduler_args(parser):
    # optimizer
    parser.add_argument('--opt', default='sgd', type=str,
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--base_lr', type=float, default=None,
                        help='base_lr if using scaling lr (lr = base_lr * bs/256')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--opt_eps', default=1e-8, type=float,
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    # fp 16 stability and gradient accumulation
    parser.add_argument('--fp16', action='store_false', help='use fp16 (on by default)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate before backward/update pass.')
    # lr scheduler
    parser.add_argument('--sched', default='cosine', type=str,
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--t_in_epochs', action='store_true',
                        help='update per epoch (instead of per iter)')
    parser.add_argument('--lr_noise', type=float, nargs='+', default=None,
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr_noise_pct', type=float, default=0.67,
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr_noise_std', type=float, default=1.0,
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6,
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay_epochs', type=float, default=30,
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='steps to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown_epochs', type=int, default=5,
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience_epochs', type=int, default=10,
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1,
                        help='LR decay rate (default: 0.1)')
    return parser


def add_vit_args(parser):
    # default config is based on vit_b16
    parser.add_argument('--patch_stride', type=int,
                        help='patch stride for patchifier, e.g. 16 for vit_b16')
    parser.add_argument('--classifier', type=str, default=None, choices=HEADS)
    parser.add_argument('--class_proj_size', type=int, default=64,
                        help='low rank projection for bilinear pooling methods')
    # auxiliary loss
    parser.add_argument('--classifier_aux', type=str, default=None, choices=AUX)
    parser.add_argument('--loss_aux_weight', type=float, default=0.1,
                        help='weight for auxiliary loss')
    parser.add_argument('--supcon', action='store_true',
                        help='if true then uses labels as in supcon when using multi_cont loss')
    parser.add_argument('--norm_ind', action='store_true',
                        help='if true then normalizes features individually when using multi_cont loss')
    # encoder related
    parser.add_argument('--attention', type=str,
                        help='use global attention or a mix of locan and global')
    parser.add_argument('--pos_embedding_type', type=str, default=None,
                        help='positional embedding for encoder, def: learned')
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--intermediate_size', type=int, default=None)
    parser.add_argument('--num_attention_heads', type=int, default=None)
    parser.add_argument('--num_hidden_layers', type=int, default=None)
    parser.add_argument('--encoder_norm', action='store_false',
                        help='norm after encoder (def: true)')
    # transformers in general
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=None)
    parser.add_argument('--hidden_dropout_prob', type=float, default=None)
    parser.add_argument('--layer_norm_eps', type=float, default=None)
    parser.add_argument('--hidden_act', type=str, default=None)
    return parser

def add_fgir_args(parser):
    # glsim
    parser.add_argument('--selector', type=str, default=None,
                        choices=('glsim', 'rollout', 'maws', 'ffvt', 'cal', 'sil'),
                        help='discriminative feature selection mechanism')
    parser.add_argument('--selector_num_tokens', type=int, default=0,
                        choices='num tokens to use for selectors')
    parser.add_argument('--crop_mode', type=str, choices=('crop_mask', 'crop', 'mask'),
                        help='crop, mask or both when using two passes')
    parser.add_argument('--dynamic_anchor', action='store_true',
                        help='use anchors (dynamic)')
    parser.add_argument('--dynamic_top', type=int, default=8,
                        help='top regions for calculating bounding boxes in dynamic anchor')
    parser.add_argument('--anchor_random', action='store_true',
                        help='if true then uses a random anchor instead of similarity')
    parser.add_argument('--sim_metric', type=str, default=None,
                        choices=SIM_METRICS, help='similarity metric for crops',)

    parser.add_argument('--anchor_class_token', action='store_true',
                        help='if true then uses its own class token for anchors instead of sharing')
    # reducer
    parser.add_argument('--reducer', type=str, choices=REDUCERS, default=None,
                        help='design of reducer module')
    # aggregator
    parser.add_argument('--representation_size', type=int, default=None)
    parser.add_argument('--aggregator', action='store_true',
                        help='use a second encoder after reducer')
    parser.add_argument('--aggregator_num_hidden_layers', type=int, default=None)

    # ifahead
    parser.add_argument('--ifa_num_layers', type=int, default=None)
    parser.add_argument('--ifa_dws_conv_groups', type=int, default=4)
    parser.add_argument('--ifa_norm', type=str, default='bn')

    # attention data augmentation
    parser.add_argument('--attention_crop', action='store_true')
    parser.add_argument('--attention_pool_size', type=int, default=1)
    parser.add_argument('--crop_attention_th', type=float, default=0.5)
    parser.add_argument('--crop_th_max', action='store_false')

    # frozen backbone
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--prompt_len', type=int, default=100)

    parser.add_argument('--adapter', type=str, default=None,
                        help='adapter module inside transformer blocks')
    parser.add_argument('--adapter_dim', type=int, default=None,
                        help='adapter module inside transformer blocks')
    parser.add_argument('--adapter_kernel_size', type=int, default=3)
    parser.add_argument('--adapter_path', type=str, default='convpass')

    parser.add_argument('--ila', action='store_true')
    parser.add_argument('--ila_locs', type=int, nargs='*', default=[-1])
    parser.add_argument('--ila_ds_locs', type=int, nargs='*', default=[3, 7])
    parser.add_argument('--ila_cls_conv', action='store_false',
                        help='separate adapter for cls token')
    parser.add_argument('--ila_ds_conv', action='store_false')
    parser.add_argument('--ila_ds_conv_type', type=str, default='dws_near_ones_init',
                        choices=('dws_near_ones_init', 'dws_ones_init', 'dws', 'conv', 'avg_pool'))
    parser.add_argument('--ila_ds_kernel_size', type=int, default=3)
    parser.add_argument('--ila_norm1', type=str, default='bn')
    parser.add_argument('--ila_norm2', type=str, default='bn')
    parser.add_argument('--ila_norm3', type=str, default='bn')
    parser.add_argument('--ila_sd', type=float, default=0.0)
    parser.add_argument('--ila_padding', action='store_true')
    parser.add_argument('--ila_dilation', type=int, default=1)
    parser.add_argument('--ila_dws_conv_groups', type=int, default=1)

    parser.add_argument('--unfreeze_cls', action='store_false')
    parser.add_argument('--unfreeze_positional_embedding', action='store_true')
    parser.add_argument('--unfreeze_mhsa_proj', action='store_true')
    parser.add_argument('--unfreeze_encoder_norm', action='store_true')
    parser.add_argument('--unfreeze_encoder_last_block', action='store_true')

    return parser


def add_augmentation_args(parser):
    # cropping
    parser.add_argument('--resize_size', type=int, default=None, help='resize_size before cropping')
    parser.add_argument('--test_resize_size', type=int, default=None, help='test resize_size before cropping')
    parser.add_argument('--test_resize_directly', action='store_true',
                        help='resizes directly to target image size instead of to larger then center crop')
    parser.add_argument('--train_resize_directly', action='store_true')
    parser.add_argument('--random_resized_crop', action='store_true',
                        help='crop random aspect ratio then resize to square')
    parser.add_argument('--square_resize_random_crop', action='store_true',
                        help='resize first to square then random crop')
    parser.add_argument('--short_side_resize_random_crop', action='store_true',
                        help='resize first so short side is resize_size then random crop a square')
    parser.add_argument('--square_center_crop', action='store_true',
                        help='resize first to square then center crop when training')
    # https://github.com/ArdhenduBehera/cap/blob/main/image_datagenerator.py
    parser.add_argument('--affine', action='store_true',
                        help='affine transform as in CAP')
    # flips
    parser.add_argument('--horizontal_flip', action='store_true',
                        help='use horizontal flip when training (on by default)')
    parser.add_argument('--vertical_flip', action='store_true',
                        help='use vertical flip (off by default)')
    # augmentation policies
    parser.add_argument('--auto_aug', action='store_true', help='Auto augmentation used')
    parser.add_argument('--rand_aug', action='store_true', help='RandAugment augmentation used')
    parser.add_argument('--trivial_aug', action='store_true', help='use trivialaugmentwide')
    # color and distortion
    parser.add_argument('--jitter_prob', type=float, default=0.0,
                        help='color jitter probability of applying (0.8 for simclr)')
    parser.add_argument('--jitter_bcs', type=float, default=0.4,
                        help='color jitter brightness contrast saturation (0.4 for simclr)')
    parser.add_argument('--jitter_hue', type=float, default=0.1,
                        help='color jitter hue value (0.1 for simclr)')
    parser.add_argument('--blur', type=float, default=0.0,
                        help='gaussian blur probability (0.5 for simclr)')
    parser.add_argument('--greyscale', type=float, default=0.0,
                        help='gaussian blur probability (0.2 for simclr)')
    parser.add_argument('--solarize_prob', type=float, default=0.0,
                        help='solarize transform probability (0.2 for byol if image_size>32)')
    parser.add_argument('--solarize', type=int, default=128,
                        help='solarize pixels with higher value than (def: 128)')
    # cutmix, mixup
    parser.add_argument('--cm', action='store_true', help='Use Cutmix')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='hyperparameter beta (default: 1)')
    parser.add_argument('--mu', action='store_true', help='Use Mixup')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--mix_prob', default=0.5, type=float,
                        help='mixup probability')
    # random erasing
    parser.add_argument('--re_mult', type=int, default=0)
    parser.add_argument('--re', default=0.0, type=float,
                        help='Random Erasing probability (def: 0.25)')
    parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')
    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')
    # regularization
    parser.add_argument('--ra', type=int, default=0, help='repeated augmentation (def: 3)')
    parser.add_argument('--sd', default=0.0, type=float,
                        help='rate of stochastic depth (def: 0.1)')
    parser.add_argument('--ls', action='store_true', help='label smoothing')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--focal_gamma', type=float, default=0.0,
                        help='gamma in focal loss (def: 2.0)')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='alpha in focal loss')
    return parser


def parse_train_args(ret_parser=False):
    parser = add_common_args()
    parser.add_argument('--save_images', type=int, default=10000,
                        help='save images every x iterations')
    parser.add_argument('--train_trainval', action='store_false',
                        help='when true uses trainval for train and evaluates on test \
                        otherwise use train for train and evaluates on val')
    parser.add_argument('--test_only', action='store_true',
                        help='when true skips training and evaluates model on test dataloader')
    parser.add_argument('--test_only_bs1', action='store_true',
                        help='same as test_only but forces bs 1 to emulate streaming/on-demand classification')
    parser.add_argument('--offline', action='store_true',
                        help='do not upload results to wandb')
    parser.add_argument('--test_multiple', type=int, default=5,
                        help='test multiple loops (to reduce model loading time influence)')
    parser.add_argument('--vis_errors', action='store_true',
                        help='when true shows prediction errors (turns on test_only by def)')
    parser.add_argument('--vis_errors_save', action='store_true',
                        help='when true saves prediction errors (turns on test_only by def)')
    parser.add_argument('--eval_freq', type=int, default=200, help='eval every x epochs')
    parser.add_argument('--top_k', type=int, default=2, help='for top-k acc metric')
    # models in general
    parser.add_argument('--model_name', type=str, default='vit_b16')
    parser.add_argument('--pretrained', action='store_true', help='pretrained model on imagenet')
    parser.add_argument('--ckpt_path', type=str, default=None, help='path to custom pretrained ckpt')
    parser.add_argument('--transfer_learning', action='store_true',
                        help='not load fc layer when using custom ckpt')
    parser.add_argument('--cal_single_crop', action='store_true',
                        help='single crop for cal')
    # distributed
    parser.add_argument('--dist_eval', action='store_true',
                        help='validate using dist sampler (else do it on one gpu)')
    parser = add_data_args(parser)
    parser = add_optim_scheduler_args(parser)
    parser = add_augmentation_args(parser)
    parser = add_vit_args(parser)
    parser = add_fgir_args(parser)
    parser.add_argument("--cfg", type=str,
                        help="If using it overwrites args and reads yaml file in given path")
    parser.add_argument("--cfg_method", type=str,
                        help="If using it overwrites args and reads yaml file in given path")

    if ret_parser:
        return parser

    args = parser.parse_args()
    adjust_config(args)
    args = add_adjust_common_dependent(args)

    return args


def parse_inference_args():
    parser = parse_train_args(ret_parser=True)
    parser.add_argument('--images_path', type=str, default='samples',
                        help='path to folder (with images) or image')
    parser.add_argument('--results_inference', type=str, default='results_inference',
                        help='path to folder to save result crops')

    # visualization
    parser.add_argument('--save_crops_only', action='store_false',
                        help='save only crop')
    parser.add_argument('--vis_cols', type=int, default=None,
                        help='how many columns when visualizing images')
    parser.add_argument('--vis_dataset_all', action='store_true',
                        help='if true then visualizes whole dataset otherwise first batch')
    parser.add_argument('--vis_mask_color', action='store_false',
                        help='if true then uses color heat map otherwise attn (white/black) map')
    parser.add_argument('--vis_th_topk', action='store_true',
                        help='for heatmaps use threshold of top-k largest')
    parser.add_argument('--vis_mask_pow', action='store_true',
                        help='square masks when applying heatmap')
    parser.add_argument('--vis_mask', type=str, default=None,
                        help='''which layer/mechanism to visualize:
                        glsim_norm, glsim_l, maws_l, attention_l where l is the layer (0 to L)
                        rollout_l until what layer or psm_h which head
                        ''')
    parser.add_argument('--vis_mask_all', action='store_true',
                        help='if true then visualizes all choices in vis_list')
    parser.add_argument('--vis_mask_list', type=str, nargs='+', default=VIS_LIST,
                        help='visualize a few selected methods')

    # feature metrics
    parser.add_argument('--compute_attention_average', action='store_true',
                        help='otherwise by def computes cka/l2/distances')
    parser.add_argument('--compute_attention_cka', action='store_true',
                        help='otherwise by def computes norm2 output features')

    args = parser.parse_args()
    adjust_config(args)
    args = add_adjust_common_dependent(args)
    return args


def parse_ffcv_prepare_dataset_args():
    parser = add_common_args()
    parser = add_data_args(parser)
    parser.add_argument('--ffcv_max_resolution', type=int, default=600,
                        help='max resolution for saving images')
    parser.add_argument('--ffcv_jpeg_quality', type=int, default=90,
                        help='jpeg quality')
    parser.add_argument('--ffcv_compress_probability', type=float, default=0.50,
                        help='compression prob')
    args = parser.parse_args()
    return args


def adjust_config(args):
    if args.cfg:
        config = yaml_config_hook(os.path.abspath(args.cfg))
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)

    if args.cfg_method:
        config = yaml_config_hook(os.path.abspath(args.cfg_method))
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)


if __name__ == '__main__':
    args = parse_train_args()
    print(args)
