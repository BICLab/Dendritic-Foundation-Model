import argparse
import os
from pathlib import Path
import utils
import torch
from torch import nn
from module.model import FC_att_layer4 as FC
from dataloader.ssc_dataset_f import my_Dataset
from torch import optim
import wandb
import numpy as np
from torch.utils import data


def get_args():
    parser = argparse.ArgumentParser("pmdend")
    parser.add_argument('--dataset', type=str, default='shd', help='[ssc, shd]')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', type=int, default= 64, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adamw', help='[sgd, adam, adamw]')
    parser.add_argument('--scheduler', type=str, default='cosine', help='[step, cosine, warmupcos]')
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='learnng rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--gamma', type=float, default=0.8, help='default:0.8, 0.5')
    parser.add_argument('--step_size', type=int, default= 15, help='defult:10')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--channel_size', default=128, type=int, help='hidden channel size')
    parser.add_argument('--data-path', type=Path, required=True, help='directory containing generated SHD NumPy arrays')
    parser.add_argument('--block', default='self-att', type=str, help='[self-att,mlp,mamba-mat,mamba-vec,hgrn1,hgrn2,sparsela_v,sparsela_m,Hyena]; spla1d/spla2d remain accepted')
    parser.add_argument('--wandb-project', default='dendritic-shd', help='Weights & Biases project name')
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='offline')


    args = parser.parse_args()
    print(args)

    return args

def main():
    print(os.getpid())
    args = get_args()
    block = args.block
    str_name = str(block)+"_dp0.1_bs128-adam-cosine-lr1e-2-decay0.1_"+str(args.seed)
    wandb.init(project=args.wandb_project, name=str_name, mode=args.wandb_mode)

    required_files = {
        name: args.data_path / name
        for name in ("trainX_4ms.npy", "trainY_4ms.npy", "testX_4ms.npy", "testY_4ms.npy")
    }
    missing = [str(path) for path in required_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing generated SHD files: {', '.join(missing)}")

    train_X = np.load(required_files["trainX_4ms.npy"])
    train_y = np.load(required_files["trainY_4ms.npy"]).astype(float)
    test_X = np.load(required_files["testX_4ms.npy"])
    test_y = np.load(required_files["testY_4ms.npy"]).astype(float)

    print('dataset shape: ', train_X.shape)
    print('dataset shape: ', test_X.shape)

    tensor_trainX = torch.Tensor(train_X)  # transform to torch tensor
    tensor_trainY = torch.Tensor(train_y).long()
    train_dataset = data.TensorDataset(tensor_trainX, tensor_trainY)

    tensor_testX = torch.Tensor(test_X)  # transform to torch tensor
    tensor_testY = torch.Tensor(test_y).long()
    test_dataset = data.TensorDataset(tensor_testX, tensor_testY)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                            shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                            shuffle=False, pin_memory=True) 
    
    model = FC(in_features=700, 
               channel_size=args.channel_size, 
               num_classes=20,block=args.block).cuda()
    criterion = nn.CrossEntropyLoss()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print('para')
    print(total_params/1024/1024)

    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params=model.parameters(),
                              lr=args.learning_rate, 
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params=model.parameters(),
                               lr=args.learning_rate)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(params=model.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
        
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                              step_size=args.step_size,
                                              gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         T_max=int(args.epochs),
                                                         eta_min=args.learning_rate*0.01)
    
    best_acc = .0
    acc_list = []
    loss_list = []
    for epoch in range(args.epochs):
        model.train()
        loss = utils.train(model, train_loader, criterion, optimizer, scheduler)
        print(f'epoch:{epoch}, loss:{loss:.6f}')
        model.eval()
        acc = utils.test(model, val_loader)
        if acc > best_acc:
            best_acc = acc
        print(f'epoch:{epoch}, loss:{loss:.6f}, acc:{acc:.2f}, best_acc:{best_acc:.2f}')
        
        acc_list.append(acc)
        loss_list.append(loss)
        wandb.log({                                                                                                
        "lr":scheduler.get_last_lr()[0],
        "Train Loss":loss,
        "Test Accuracy": acc,
        "Best acc": best_acc})
    

if __name__ == '__main__':
    main()
