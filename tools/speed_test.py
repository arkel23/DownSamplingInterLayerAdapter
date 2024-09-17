import time
from contextlib import suppress

import wandb
import torch
import torch.backends.cudnn as cudnn

from fgir_vit.other_utils.build_args import parse_inference_args
from fgir_vit.model_utils.build_model import build_model
from fgir_vit.data_utils.build_dataloaders import build_dataloaders
from fgir_vit.train_utils.misc_utils import set_random_seed, count_params_single, count_params_trainable, count_flops


def adjust_args_general(args):
    ila = '_ila' if args.ila else ''
    if ila:
        ila = ila if args.ila_locs else f'{ila}_dso'
    classifier = f'_{args.classifier}' if args.classifier else ''
    selector = f'_{args.selector}' if args.selector else ''
    adapter = f'_{args.adapter}' if args.adapter else ''
    prompt = f'_{args.prompt}' if args.prompt else ''
    freeze = '_fz' if args.freeze_backbone else ''

    args.run_name = '{}_{}{}{}{}{}{}{}_{}'.format(
        args.dataset_name, args.model_name, ila, classifier, selector, adapter,
        prompt, freeze, args.serial
    )

    return args

def setup_environment(args):
    set_random_seed(args.seed, numpy=True)
    cudnn.benchmark = True

    _, _, test_loader = build_dataloaders(args)

    model = build_model(args)
    model.eval()

    args = adjust_args_general(args)

    if not args.debugging:
        wandb.init(config=args, project=args.project_name, entity=args.entity)
        wandb.run.name = args.run_name

    return test_loader, model


def measure_tp(model, loader, device='cuda', multiple=1, amp=True):
    torch.cuda.synchronize()
    start = time.time()

    amp_autocast = torch.cuda.amp.autocast if amp else suppress

    for m in range(multiple):
        for i,(x, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                with amp_autocast():
                    model(x)

            if i % 100 == 0:
                print(f'{m} / {multiple}: {i} / {len(loader)}')

    torch.cuda.synchronize()
    time_total = time.time() - start

    num_images = len(loader.dataset) * multiple
    throughput = round((num_images / time_total), 4)
    time_total = round((time_total / 60), 4)
    return throughput, time_total


def main():
    args = parse_inference_args()

    test_loader, model = setup_environment(args)

    tp, time_total = measure_tp(model, test_loader, args.device, args.test_multiple, args.fp16)

    flops = count_flops(model, args.image_size, args.device)
    max_memory = round(torch.cuda.max_memory_reserved() / (1024 ** 3), 4)
    no_params = round((count_params_single(model) / 1e6), 4)
    no_params_trainable = round((count_params_trainable(model) / 1e6), 4)

    if not args.debugging:
        wandb.run.summary['throughput'] = tp
        wandb.run.summary['time_total'] = time_total
        wandb.run.summary['flops'] = flops
        wandb.run.summary['max_memory'] = max_memory
        wandb.run.summary['no_params'] = no_params
        wandb.run.summary['no_params_trainable'] = no_params_trainable
        wandb.finish()

    print('run_name,tp,time_total,flops,max_memory,no_params,no_params_trainable')
    line = f'{args.run_name},{tp},{time_total},{flops},{max_memory},{no_params},{no_params_trainable}'
    print(line)
    return 0


if __name__ == "__main__":
    main()
