import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())



import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR

# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./ds/classification/seven_class/train',
                    help='path to dataset') ## This one should be changed to only include training split?
parser.add_argument('-dataset-name', default='train',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    # choices=model_names,
                    # help='model architecture: ' +
                    #      ' | '.join(model_names) +
                    #      ' (default: resnet50)'
                    )
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                     help='number of data loading workers (default: 32)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=32, type=int,
                    help='feature dimension (default: 128)')
# parser.add_argument('--log-every-n-steps', default=100, type=int,
#                     help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
# parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--device', default='cuda:0', type=str, help='Gpu index.')

parser.add_argument('--sam', action='store_true',
                    help='Whether or not to use sharpness sware minimization.')
parser.add_argument('--val-open', dest='val_open', action='store_true', default=False,
                    help='Enable held-out contrastive validation (val-half, every 10 epochs). '
                         'When on, ckpts get a _wval suffix (+ a _best.pkl at lowest val-loss) so '
                         'the original no-val ckpts are never overwritten. Default off.')
parser.add_argument('--grad_cache', default='auto', choices=['auto', 'on', 'off'],
                    help="Use GradCache training. 'auto' (default): resnet18 → plain train(), "
                         "larger archs → GradCache. 'off': force the plain train() path for ALL "
                         "archs (identical to resnet18; will need enough VRAM for full bs). "
                         "'on': force GradCache for all archs.")
parser.add_argument('--micro_bs', default=64, type=int,
                    help='GradCache micro-batch (per-view rows), used only when GradCache is active. '
                         'Pick the largest that fits to minimize the BatchNorm-per-microbatch gap.')

def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    # if not args.disable_cuda and torch.cuda.is_available():
    #     args.device = torch.device('cuda')
    #     cudnn.deterministic = True
    #     cudnn.benchmark = True
    # else:
    #     args.device = torch.device('cpu')
    #     args.gpu_index = -1

    dataset = ContrastiveLearningDataset(args.data)

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # held-out contrastive validation — ONLY when --val-open (else original no-val behaviour).
    # Same val-half as classification get_data; one big batch (~254 imgs) so InfoNCE negatives
    # ≈ training's. num_workers=0 → RNG snapshot in simclr._run_validation fully controls it.
    val_loader = None
    if args.val_open:
        val_dataset = dataset.get_val_dataset(args.n_views)
        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=4096, shuffle=False, num_workers=0,
                pin_memory=True, drop_last=False)
            print(f"Contrastive validation ON: {len(val_dataset)} images (val-half), one batch.")
        else:
            print("[warn] --val-open set but no val/ folder found → running without validation.")

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    ## This is wrong!
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
    #                                                        last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.device):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        # GradCache routing (--grad_cache auto|on|off):
        #   auto → resnet18 plain train(); larger archs GradCache (default).
        #   off  → plain train() for ALL archs (identical path to resnet18; needs full-bs VRAM).
        #   on   → GradCache for ALL archs.
        # resnet18's plain train() is byte-for-byte untouched, so it always reproduces.
        is_resnet18 = args.arch in ("resnet18", "resnet18_random")
        use_gradcache = (args.grad_cache == 'on') or (args.grad_cache == 'auto' and not is_resnet18)
        if args.sam:
            print('Train with Sharpness aware minimization!')
            simclr.train_with_sam(train_loader)
        elif use_gradcache:
            print(f'GradCache Training (arch={args.arch}, micro_bs={args.micro_bs})!')
            simclr.train_gradcache(train_loader, micro_bs=args.micro_bs, val_loader=val_loader)
        else:
            print(f'Normal Training (no GradCache, arch={args.arch})!')
            simclr.train(train_loader, val_loader=val_loader)


if __name__ == "__main__":
    main()