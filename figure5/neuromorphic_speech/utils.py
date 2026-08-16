import torch
import torch.nn.functional as F
import numpy as np
import random as python_random
import os
import json
import sys
from tqdm import tqdm

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    np.random.seed(seed)
    
    python_random.seed(seed)
    # cuda env
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    print('seeds are fixed')

def train(model, train_loader, criterion, optimizer, scheduler=None):
    train_loss = 0
    correct = 0
    model.train()
    for (imgs, targets) in tqdm(train_loader):
        train_loss = 0.0
        optimizer.zero_grad()
        imgs, targets = imgs.cuda(), targets.cuda().squeeze(-1)
        output = model(imgs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        train_loss = train_loss + loss.item()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if len(targets.shape) == 2:
            targets = targets.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum().item()
    
    train_loss = train_loss * targets.shape[0] / len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    del output
    torch.cuda.empty_cache()
    if scheduler is not None:
        scheduler.step()
    
    print(f'Train_Accuracy: {accuracy:2f}')
        
    return train_loss

def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for (data, target) in tqdm(test_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
        del output
        torch.cuda.empty_cache()
    return accuracy

def dump_json(obj, fdir, name):
    """
    Dump python object in json
    """
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(os.path.join(fdir, name), "w") as f:
        json.dump(obj, f, indent=4, sort_keys=False)

def load_json(fdir, name):
    """
    Load json as python object
    """
    path = os.path.join(fdir, name)
    if not os.path.exists(path):
        raise FileNotFoundError("Could not find json file: {}".format(path))
    with open(path, "r") as f:
        obj = json.load(f)
    return obj 