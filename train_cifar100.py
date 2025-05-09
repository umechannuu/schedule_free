import os
import json
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from utils.schedulefree.sgd_schedulefree import SGDScheduleFree
from utils import checkpoint, get_config_value, get_lr_scheduler, save_to_csv, select_model, get_optimizer, get_bs_scheduler


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training with ScheduleFree SGD')
    parser.add_argument('config_path', type = str, help='Path to the config file(.json)')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--cuda_device', type = str, default='0', help='CUDA device to use')
    return parser.parse_args()


def train(epoch, steps, model, device, trainset, optimizer, lr_scheduler, lr_step_type, criterion, batch_size, cuda):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    print('\nEpoch: %d' % epoch)
    model.train()
    if isinstance(optimizer, SGDScheduleFree):
        optimizer.train()
    
    train_loss = 0
    correct = 0
    total = 0
    lr_batch = []

    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        steps += 1
        

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if hasattr(lr_scheduler, 'step') and lr_step_type == 'step':
            lr_scheduler.step()

        last_lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') else optimizer.param_groups[0]['scheduled_lr']
        lr_batch.append([epoch + 1, steps, last_lr, batch_size])

    p_norm = get_full_grad_list(net, trainset, optimizer, batch_size ,cuda)
    norm_result = [epoch + 1, steps, p_norm]
    training_acc = 100.*correct/total
    train_result = [epoch + 1, steps, train_loss/(batch_idx + 1), training_acc, last_lr]

    return steps, lr_batch, train_result, norm_result


def test(epoch, model, optimizer, device, testloader, criterion):
    net.eval()
    if isinstance(optimizer, SGDScheduleFree):
        optimizer.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    acc = 100.*correct/total
    test_result = [epoch + 1, test_loss/(batch_idx + 1), acc]
    return test_result


def get_full_grad_list(net, train_set, optimizer , batch_size, cuda_device):
    parameters = [p for p in net.parameters()]
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=2)
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    init=True
    full_grad_list=[]

    for i, (xx,yy) in (enumerate(train_loader)):
        xx = xx.to(device, non_blocking = True)
        yy = yy.to(device, non_blocking = True)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss(reduction='mean')(net(xx), yy)
        loss.backward()
        if init:
            for params in parameters:
                full_grad = torch.zeros_like(params.grad.detach().data)
                full_grad_list.append(full_grad)
            init=False

        for i, params in enumerate(parameters):
            g = params.grad.detach().data
            full_grad_list[i] += (batch_size / len(train_set)) * g
    
    total_norm = sum(grad.norm(2).item() ** 2 for grad in full_grad_list) ** 0.5
    return total_norm 

if __name__ == "__main__":
    args = get_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    lr = get_config_value(config, "init_lr")
    epochs = get_config_value(config, "epochs")
    checkpoint_path = config.get("checkpoint_path", "checkpoint.pth.tar")
    csv_path = get_config_value(config, "csv_path")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Data augmentation and normalization for training
    # Just normalization for validation
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # set CUDA device
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    model_name = get_config_value(config, "model_name")
    net = select_model(model_name)
    net = net.to(device)
    print(f"Using model: {model_name}")
    print(f"Using device: {device}")


    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(net.parameters(), config)
    bs_scheduler, total_steps = get_bs_scheduler(config, trainset_length=len(trainset))
    print(f"Using optimizer: {optimizer}")
    
    lr_scheduler, lr_step_type = get_lr_scheduler(optimizer, config, total_steps)
    train_results = []
    test_results = []
    norm_results = []
    lr_batches = []
    schedulefree_internal = []
    
    if args.resume:
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        steps = checkpoint['steps']
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
    else:
        start_epoch = 0
        steps = 0
        batch_size = bs_scheduler.get_batch_size()
        lr_batches.append([1, steps, lr, batch_size])

    for epoch in range(start_epoch, epochs):
        batch_size = bs_scheduler.get_batch_size()
        print(f"Batch size: {batch_size}")
        print(f"learning rate: {optimizer.param_groups[0]['lr']}") if hasattr(optimizer, 'param_groups') else optimizer.param_groups[0]['scheduled_lr']

        steps, lr_batch, train_result, norm_result = train(epoch, steps, net, device, trainset, optimizer, lr_scheduler, lr_step_type, criterion, batch_size, args.cuda_device)
        lr_batches.append(lr_batch)
        train_results.append(train_result)
        norm_results.append(norm_result)

        test_result = test(epoch, net, optimizer, device, testloader, criterion)
        test_results.append(test_result)
        schedulefree_internal.append([optimizer.ckp1.item() if hasattr(optimizer, "ckp1") else None,
                                    optimizer.ckp2.item() if hasattr(optimizer, "ckp2") else None,
                                    optimizer.kappa if hasattr(optimizer, "kappa") else None,
                                    optimizer.beta if hasattr(optimizer, "beta") else None])
        if hasattr(lr_scheduler, 'step') and lr_step_type == 'epoch':
            lr_scheduler.step()

        checkpoint.save({
            'epoch': epoch,
            'steps': steps,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict() if hasattr(lr_scheduler, 'state_dict') else {},
            'bs_scheduler_state_dict': bs_scheduler.state_dict(),
            'train_results': train_results,
            'test_results': test_results,
            'norm_results': norm_results,
            'lr_batches': lr_batches,
        }, checkpoint_path)

        if isinstance(optimizer, SGDScheduleFree):
            ckp1 = optimizer.ckp1.item() if hasattr(optimizer, "ckp1") else None
            ckp2 = optimizer.ckp2.item() if hasattr(optimizer, "ckp2") else None
            kappa = optimizer.kappa if hasattr(optimizer, "kappa") else None
            beta = optimizer.beta if hasattr(optimizer, "beta") else None
            schedulefree_internal.append([ckp1, ckp2, kappa, beta])

        print(f'Epoch: {epoch + 1}, Steps: {steps}, Train Loss: {train_results[epoch][2]:.4f}, Test Acc: {test_results[epoch][2]:.2f}%')

    # Save results to CSV
    save_to_csv(csv_path, {
        "train": train_results,
        "test": test_results,
        "norm": norm_results,
        "lr_bs": lr_batches,
        "ckp1": [item[0] if item[0] is not None else ["NA"] for item in schedulefree_internal],
        "ckp2": [item[1] if item[1] is not None else ["NA"] for item in schedulefree_internal],
        "kappa": [item[2] if item[2] is not None else ["NA"] for item in schedulefree_internal],
        "beta": [item[3] if item[3] is not None else ["NA"] for item in schedulefree_internal],
    })