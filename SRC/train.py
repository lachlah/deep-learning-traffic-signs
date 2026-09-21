from model import create_model
import torch
import torch.nn as nn
from torch.optim import Adam
def train():
    model=create_model()
    loss=0
    accuracy=0
    model.train()
    loss_function=nn.CrossEntropyLoss()
    optimizer=Adam(base_model.parameters())
    for x,y in train_loader:
        output=model(x)
        optimizer.zero_grad()
        batch_loss=loss_function(output,y)
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output,y,train_N)
    print('Valid-Loss:{:.4f} Accuracy {:.4f}'.format(loss,accuracy))