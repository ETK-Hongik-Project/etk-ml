from gtk_dataset import GTKDataset
from model import TinyTracker
import argparse
import torch
from torch.optim import *
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def parse_args():
    """
    Usage: args = parse_args_train()
    """
    my_parser = argparse.ArgumentParser(description='GTK-Trainer.')
    my_parser.add_argument('--data-path', type=str,
                           help="Path to processed dataset", required=True)
    my_parser.add_argument('--model-path', type=str,
                           help='Path to save model', required=True)
    my_parser.add_argument('--num-epochs', type=int, default=20, )
    my_parser.add_argument('--batch-size', type=int, default=64, )
    my_parser.add_argument('--image-size', type=int, default=112, )
    my_parser.add_argument('--learning-rate', type=float,
                           default=0.001, )
    my_parser.add_argument('--backbone', type=str, default="mobilenetv3",
                           help="Set path of pretrained model weights", )
    my_parser.add_argument('--in-channel', type=int, default=1,
                           help="Num of input channel", )
    return my_parser.parse_args()


def train_epoch(dataloader: DataLoader, model: nn.Module,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device):
    model.train()
    epoch_loss = 0.0
    n_batches = len(dataloader)

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / n_batches


def evaluate(dataloader: DataLoader, model: nn.Module,
             criterion: nn.Module, device: torch.device,
             return_preds: bool = False):
    model.eval()
    epoch_loss = 0.0
    n_batches = len(dataloader)
    preds = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            if return_preds:
                preds.append(pred.cpu().numpy())
            loss = criterion(pred, y)

            epoch_loss += loss.item()

    if return_preds:
        preds = np.concatenate(preds, axis=0)

    return epoch_loss / n_batches, preds


if __name__ == "__main__":
    # TODO: Train Loop
    arguments = parse_args()
    print(arguments.data_path, arguments.in_channel)

    # Set Device
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available() and \
            torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'{device} is available')

    model = TinyTracker(in_channels=arguments.in_channel,
                        backbone=arguments.backbone)

    train_set = GTKDataset(data_path=arguments.data_path, split='train',
                           img_size=arguments.image_size, channel=arguments.in_channel)
    val_set = GTKDataset(data_path=arguments.data_path, split='val',
                         img_size=arguments.image_size, channel=arguments.in_channel)
    test_set = GTKDataset(data_path=arguments.data_path, split='test',
                          img_size=arguments.image_size, channel=arguments.in_channel)

    train_loader = DataLoader(train_set,
                              batch_size=arguments.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_set,
                            batch_size=arguments.batch_size,
                            shuffle=True)
    test_loader = DataLoader(test_set,
                             batch_size=arguments.batch_size,
                             shuffle=True)

    model.to(device)
    optim = Adam(lr=arguments.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    loss_graph = []
    val_loss_graph = []
    best_loss = 987654321
    progress_bar = tqdm(range(arguments.num_epochs), desc='Epoch', leave=True)
    for epoch in progress_bar:
        train_loss = train_epoch(dataloader=train_loader, model=model,
                                 optimizer=optim, criterion=loss_fn, device=device)
        loss_graph.append(train_loss)

        if epoch % 3 == 0:
            val_loss, _ = evaluate(dataloader=val_loader, model=model,
                                   criterion=loss_fn, device=device, return_preds=False)
            val_loss_graph(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss

        progress_bar.set_postfix(
            {
                'Train Loss': f'{train_loss:.4e}',
                'Validation Loss': f'{val_loss:.4e}'
            }
        )

    test_loss = evaluate(dataloader=test_loader, model=model,
                         criterion=loss_fn, device=device, return_preds=False)
    tqdm.write(f"Model Training Done. Test Loss : {test_loss:.4f}")
