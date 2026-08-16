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

def get_args():
    parser = argparse.ArgumentParser("pmdend")
    parser.add_argument('--dataset', type=str, default='ssc', help='[ssc, shd]')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', type=int, default= 128, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adamw', help='[sgd, adam, adamw]')
    parser.add_argument('--scheduler', type=str, default='cosine', help='[step, cosine, warmupcos]')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='learnng rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--gamma', type=float, default=0.8, help='default:0.8, 0.5')
    parser.add_argument('--step_size', type=int, default= 15, help='defult:10')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--channel_size', default=128, type=int, help='hidden channel size')
    parser.add_argument('--data-path', type=Path, required=True, help='root containing train, valid, and test splits')
    parser.add_argument('--block', default='self-att', type=str, help='[self-att,mlp,mamba-mat,mamba-vec,hgrn1,hgrn2,sparsela_v,sparsela_m,Hyena]; spla1d/spla2d remain accepted')
    parser.add_argument('--wandb-project', default='dendritic-ssc', help='Weights & Biases project name')
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='offline')

    args = parser.parse_args()
    print(args)

    return args

def main():
    print(os.getpid())
    args = get_args()
    utils.set_seed(args.seed)
    block = args.block
    str_name = str(block)+"0.3m_dp0.1_bs128-adamw-cosine-lr1e-2-decay0.1_"+str(args.seed)
    wandb.init(project=args.wandb_project, name=str_name, mode=args.wandb_mode)

    train_dir = args.data_path / 'train'
    valid_dir = args.data_path / 'valid'
    test_dir = args.data_path / 'test'
    for split_dir in (train_dir, valid_dir, test_dir):
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing SSC split directory: {split_dir}")
    train_files = [str(path) for path in train_dir.iterdir()]
    valid_files = [str(path) for path in valid_dir.iterdir()]
    test_files = [str(path) for path in test_dir.iterdir()]
    train_dataset = my_Dataset(train_files)
    print(len(train_dataset))
    valid_dataset = my_Dataset(valid_files)
    test_dataset = my_Dataset(test_files)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                            shuffle=True, pin_memory=True,num_workers=16)
    val_loader  = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                            shuffle=False, pin_memory=True,num_workers=2) 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                            shuffle=False, pin_memory=True,num_workers=2) 
    
    model = FC(in_features=700, 
               channel_size=args.channel_size, 
               num_classes=35,block=args.block).cuda()
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
        
    best_acc_val = .0
    best_acc_test = .0
    acc_list = []
    loss_list = []
    for epoch in range(args.epochs):
        model.train()
        loss = utils.train(model, train_loader, criterion, optimizer, scheduler)
        print(f'epoch:{epoch}, loss:{loss:.6f}')
        model.eval()
        acc_val = utils.test(model, val_loader)
        acc_test = utils.test(model,test_loader)
        if acc_val > best_acc_val:
            best_acc_val = acc_val
            best_acc_test = acc_test
            torch.save(model.state_dict(), 'ckpt/'+str_name+'.pth')
        print(f'epoch:{epoch}, loss:{loss:.6f}, acc:{acc_val:.2f}, best_acc_val:{best_acc_val:.2f},test_acc:{acc_test:.2f},best_test_acc:{best_acc_test:.2f}')
                                                                                                                                        
        acc_list.append(acc_val)
        loss_list.append(loss)
        wandb.log({                                                                                                
        "lr":scheduler.get_last_lr()[0],
        "Train Loss":loss,
        "Val Accuracy": acc_val,
        "Best acc": best_acc_val,
        "test_acc":acc_test,})
    

if __name__ == '__main__':
    main()
