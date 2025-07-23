import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from .evaluate import evaluate 


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    ) -> Dict[str, float]:
    scaler = GradScaler()

    epoch_train_losses = []
    epoch_test_losses = []
    epoch_test_acc = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader), 0):
            # remove the label from batch
            labels = batch.pop('label').to(device)
            # dictionary of features, with features on device
            inputs = {k: v.to(device) for k, v in batch.items()}
            # zero grads
            optimizer.zero_grad()
            # forward + backward + optimize
            with autocast(device_type=device.type):
              outputs = model(inputs)
              loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # store training loss
            running_loss += loss.item()
        # calculate avg training epoch loss
        avg_epoch_loss = running_loss/len(train_loader)
        # add to running list
        epoch_train_losses.append(avg_epoch_loss)
        # get test set evaluation stats
        eval_dict = evaluate(model, test_loader, criterion, device)
        test_loss = eval_dict['loss']
        test_acc = eval_dict['accuracy']
        epoch_test_losses.append(test_loss)
        epoch_test_acc.append(test_acc)
        # print stats after each epoch
        print(f' avg epoch train loss: {round(avg_epoch_loss, 4)}\n \
        test set loss: {round(test_loss,4)}\n test set accuracy: {round(test_acc,4)}')

    print(f'Completed training for {epochs} epochs')
    return {'train_losses': epoch_train_losses, 'test_losses': epoch_test_losses, 'test_acc': epoch_test_acc}