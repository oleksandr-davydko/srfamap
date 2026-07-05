import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["RUST_BACKTRACE"] = "full"
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data.datasets import load_dataset_with_heldout_test
from radiomics_saliency import config
from radiomics_saliency.experiment import run_experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Second-order statistical radiomic features saliency maps benchmark')
    parser.add_argument('-b', '--begin', type=int)
    parser.add_argument('-e', '--end', type=int)
    parser.add_argument('-d', '--device', type=int)
    parser.add_argument('-dt', '--datapath', type=str)
    parser.add_argument('-dn', '--datasetname', type=str)
    parser.add_argument('-f', '--features', type=str)
    parser.add_argument('-a', '--alpha', type=int, default=0)
    parser.add_argument('-dl', '--delta', type=int, default=1)
    parser.add_argument('-z', '--omitzeros', type=bool, default=True)
    parser.add_argument('-r', '--resultdir', type=str)
    parser.add_argument('-st', '--skiptraining', type=bool, default=False)
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('--saliency-batch-size', type=int, default=4)
    parser.add_argument('--saliency-n-steps', type=int, default=32)
    parser.add_argument('--attribution-mode', type=str, default='pairwise',
                        choices=['pairwise', 'ovr_margin', 'logit'],
                        help="Multi-class IG target: pairwise (predicted vs runner-up), "
                             "ovr_margin (predicted vs rest), or logit (raw class logit).")
    parser.add_argument('--eg-samples', type=int, default=24,
                        help='Number of Expected-Gradients baselines averaged per attribution.')
    parser.add_argument('--eg-baseline-pool', type=int, default=64,
                        help='Number of training images sampled to form the EG baseline pool.')
    parser.add_argument('--completeness-sample', type=int, default=16,
                        help='Number of test images used for the completeness diagnostic (0 to skip).')
    parser.add_argument('--profile-saliency', action='store_true')
    args = parser.parse_args()
    model_name = f'{args.features}'
    config.device = torch.device(f'cuda:{args.device}')
    train_dev_images, train_dev_labels, heldout_test_images, heldout_test_labels, heldout_test_ids = \
        load_dataset_with_heldout_test(args.datasetname, args.datapath)
    print(train_dev_images.shape)
    print(train_dev_labels.shape)
    print(heldout_test_images.shape)
    print(heldout_test_labels.shape)
    config.image_max_width = train_dev_images.shape[1]
    dev_size_within_train_dev = 15 / 85

    # Create one held-out test set, then resplit only the train/dev pool per iteration.
    for iteration in range(args.begin, args.end):
        train_dev_indices = np.arange(train_dev_images.shape[0])
        indices_train, indices_dev, _, _ = train_test_split(
            train_dev_indices, train_dev_labels,
            test_size=dev_size_within_train_dev,
            random_state=iteration,
            stratify=train_dev_labels.reshape(-1)
        )

        result_dir = f'{args.resultdir}{model_name}/{iteration}'
        os.makedirs(result_dir, exist_ok=True)

        run_experiment(train_dev_images, train_dev_labels, indices_train, indices_dev,
                       heldout_test_images, heldout_test_labels, config.device,
                       result_dir, test_ids=heldout_test_ids,
                       alpha=args.alpha, delta=args.delta, omit_zeros=args.omitzeros,
                       skip_training=args.skiptraining, model_type_name=args.model,
                       saliency_batch_size=args.saliency_batch_size,
                       saliency_n_steps=args.saliency_n_steps,
                       attribution_mode=args.attribution_mode,
                       eg_samples=args.eg_samples,
                       eg_baseline_pool=args.eg_baseline_pool,
                       completeness_sample=args.completeness_sample,
                       profile_saliency=args.profile_saliency)
