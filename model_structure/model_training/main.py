from SELFRec import SELFRec
from util.conf import ModelConf
import time
import os
from paras import args
import argparse


def print_models(title, models):
    print(f"{'=' * 80}\n{title}\n{'-' * 80}")
    for category, model_list in models.items():
        print(f"{category}:\n   {'   '.join(model_list)}\n{'-' * 80}")


if __name__ == '__main__':
    # Define available models
    models = {
        'Graph-Based Baseline Models': ['LightGCN', 'DirectAU', 'MF', 'NGCF','DGCF'],
        'Self-Supervised Graph-Based Models': ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL',
                                               'NCL', 'MixGCF', 'SCCF'],
        'Sequential Baseline Models': ['SASRec'],
        'Self-Supervised Sequential Models': ['CL4SRec', 'BERT4Rec']
    }

    # Parse arguments


    # Set GPU device
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("Using CPU")

    # Print available models
    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print_models("Available Models", models)

    # Check if the model exists
    all_models = sum(models.values(), [])
    if args.model in all_models:
        s = time.time()
        conf = ModelConf(f'./conf/{args.model}.yaml')
        rec = SELFRec(conf)
        rec.execute()
        e = time.time()
        print(f"Running time: {e - s:.2f} s")
    else:
        print('Wrong model name!')
        exit(-1)
