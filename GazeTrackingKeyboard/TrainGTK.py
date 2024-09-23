from Dataset import GTKDataset
import argparse
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm


def parse_args():
    """
    Usage: args = parse_args_train()
    """
    my_parser = argparse.ArgumentParser(description='GTK-Trainer.')
    my_parser.add_argument('--data-path', type=str,
                           help="Path to processed dataset", metavar=' ')
    my_parser.add_argument('--num-epochs', type=int, default=20, metavar=' ')
    my_parser.add_argument('--batch-size', type=int, default=64, metavar=' ')
    my_parser.add_argument('--image-size', type=int, default=112, metavar=' ')
    my_parser.add_argument('--learning-rate', type=float,
                           default=0.001, metavar=' ')
    my_parser.add_argument('--backbone', type=str, default="mobilenetv3",
                           help="Set path of pretrained model weights", metavar=' ')
    my_parser.add_argument('--in-channel', type=int, default=1,
                           help="Num of input channel", metavar=' ')
    return my_parser.parse_args()


if __name__ == "__main__":
    # TODO: Train Loop
