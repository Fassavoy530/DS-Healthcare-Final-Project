import argparse
from utils import prepare_save_dir
from STELLAR import STELLAR
import numpy as np
import os
import torch
from datasets import GraphDataset, load_mouselymph_data

#RUN MouseLymphCross COMMAND:
# python run.py --dataset MouseLymphCross --num-heads 14 --num-seed-class 3 --wd 0 --lr 1e-3 --batch-size 32 --epoch 320 --distance_thres 50
#RUN BETonsil COMMAND:
# python run.py --dataset TonsilBE --num-heads 13 --num-seed-class 3 --wd 0 --lr 1e-3 --batch-size 32 --epoch 320 --distance_thres 30


def main():
    parser = argparse.ArgumentParser(description='STELLAR')
    parser.add_argument('--dataset', default='Hubmap', help='dataset setting')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='STELLAR')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--num-heads', type=int, default=22)
    parser.add_argument('--num-seed-class', type=int, default=0)
    parser.add_argument('--sample-rate', type=float, default=1)
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N',
                        help='mini-batch size')
    parser.add_argument('--distance_thres', default=50, type=int)
    parser.add_argument('--savedir', type=str, default='./')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Seed the run and create saving directory
    args.name = '_'.join([args.dataset, args.name])
    args = prepare_save_dir(args, __file__)

    if args.dataset == 'MouseLymph':
        labeled_X, labeled_y, unlabeled_X, test_y, labeled_edges, unlabeled_edges, inverse_dict = load_mouselymph_data(
            './data/mouse_lymph_simplified.csv', args.distance_thres, args.sample_rate, way = 'strat')
        dataset = GraphDataset(labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges)
    elif args.dataset == 'MouseLymphCross':
        labeled_X, labeled_y, unlabeled_X, test_y, labeled_edges, unlabeled_edges, inverse_dict = load_mouselymph_data(
            './data/region1_2_healthy_mouse_lymph_simplified.csv', args.distance_thres, args.sample_rate, way = 'cross')
        dataset = GraphDataset(labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges)
    elif args.dataset == 'MouseLymphCrossInfection':
        labeled_X, labeled_y, unlabeled_X, test_y, labeled_edges, unlabeled_edges, inverse_dict = load_mouselymph_data(
            './data/region1_healthy_infected_mouse_simplified_revised.csv', args.distance_thres, args.sample_rate, way = 'cross_infection')
        dataset = GraphDataset(labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges)
    stellar = STELLAR(args, dataset)
    stellar.train()
    _, results = stellar.pred()
    np.save(os.path.join(args.savedir, args.dataset + '_results__betonsil_epoch320_batch32_dist30_novel.npy'), results)


if __name__ == '__main__':
    main()