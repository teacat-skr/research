import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms;
from sklearn.metrics import classification_report
import math
from model import resnet18k

import argparse
import random
import matplotlib.pyplot as plt
import csv 
import warnings

def main():
    epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True, num_workers=2)
    class_names = ('plane', 'car', 'bird', 'cat', 'dog', 'frog', 'ship', 'truck')
    model = torchvision.models.resnet18(pretrained=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    x = []
    tracc = []
    teacc = []
    x.append(0)
    tracc.append(0)
    teacc.append(0)
    data_save = []
    for epoch in range(epochs):
        train_acc, train_loss = train(model, device, train_loader, criterion, optimizer)
        test_acc, test_loss = test(model, device, test_loader, criterion)
        stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, test acc: {:<8}, test loss: {:<8}'
        print(stdout_temp.format(epoch+1, train_acc, train_loss, test_acc, test_loss))
        print('')
        data_save.append([epoch,test_acc])
        x.append(epoch + 1)
        tracc.append(train_acc)
        teacc.append(test_acc)

    
    with open('resnet18-cifat10.csv','w') as file:
        writer = csv.writer(file)
        writer.writerows(data_save)
    
    plt.title("ResNet18 trained by Cifar10")
    plt.xlim(0, epochs * 1.2)
    plt.ylim(0, 1)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(x, tracc, label = 'train')
    plt.plot(x, teacc, label = 'test')
    plt.legend(loc='upper right')
    plt.savefig("sample.png")
    
def sub():
    args = parse_args()
    #step数指定
    grad_steps = args.grad_steps
    label_noise_rate = args.label_noise_rate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    #インスタンス変数にアクセスしてラベルの張替え
    for i in range(len(train_set.targets)):
        if(random.randint(0, 9999) < int(label_noise_rate * 10000)):
            train_set.targets[i] += random.randint(1, 9)
            train_set.targets[i] %= 10
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True, num_workers=2)
    class_names = ('plane', 'car', 'bird', 'cat', 'dog', 'frog', 'ship', 'truck')
    model = resnet18k.make_resnet18k(k=args.model_width)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    def func(steps):
        if steps < 512:
            return 1
        else:
            return 1.0 / math.sqrt(math.floor(steps / 512.0))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = func)
    x1 = range(grad_steps + 1)
    x2 = []
    trerr = []
    teerr = []
    trerr.append(1)
    teerr.append(1)
    x2.append(0)
    data_save = []
    count = 0
    while(grad_steps != count):
        running_loss = 0.0
        model.train()
        for batich_idx, (inputs, targets) in enumerate(train_loader):
            if(grad_steps == count):
                break
            output_list = []
            target_list = []
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output_list += [int(o.argmax()) for o in outputs]
            target_list += [int(t) for t in targets]
            running_loss += loss.item()

            train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)
            if count % 100 == 0 and count != 0:
                stdout_temp = 'step: {:>3}/{:<3}, train acc:{:<8}, train loss: {:<8}'
                print(stdout_temp.format(count, grad_steps, train_acc, train_loss))
            trerr.append(1.0 - train_acc)
            count += 1
            scheduler.step()
        test_acc, test_loss = test(model, device, test_loader, criterion)
        stdout_temp = 'step: {:>3}/{:<3}, test acc:{:<8}, test loss: {:<8}'
        print(stdout_temp.format(count, grad_steps, test_acc, test_loss))
        teerr.append(1.0 - test_acc)
        x2.append(count)
    
    # with open('resnet18-cifat10.csv','w') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(data_save)
    
    plt.title("ResNet18 trained by Cifar10")
    plt.xlim(0, grad_steps * 1.2)
    plt.ylim(0, 1)
    plt.xlabel("Steps")
    plt.ylabel("Erorr")
    plt.plot(x1, trerr, label = 'train', linewidth=0.5)
    plt.plot(x2, teerr, label = 'test')
    plt.legend(loc='upper right')
    plt.savefig("sample1.png")


    

def train (model, device, train_loader, criterion, optimizer):
    model.train()
    output_list = []
    target_list = []
    running_loss = 0.0
    for batich_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output_list += [int(o.argmax()) for o in outputs]
        target_list += [int(t) for t in targets]
        running_loss += loss.item()

        train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)
        if batich_idx % 10 == 0 and batich_idx != 0:
            stdout_temp = 'batch: {:>3}/{:<3}, train acc:{:<8}, train loss: {:<8}'
            print(stdout_temp.format(batich_idx, len(train_loader), train_acc, train_loss))
    train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)


    return train_acc, train_loss

def test(model, device, test_loader, criterion):
	model.eval()

	output_list = []
	target_list = []
	running_loss = 0.0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(test_loader):
			# Forward processing.
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			
			# Set data to calculate score.
			output_list += [int(o.argmax()) for o in outputs]
			target_list += [int(t) for t in targets]
			running_loss += loss.item()
		
	test_acc, test_loss = calc_score(output_list, target_list, running_loss, test_loader)

	return test_acc, test_loss
def calc_score(output_list, target_list, running_loss, data_loader):
    # import pdb;pdb.set_trace()
    result = classification_report(output_list, target_list, output_dict=True)
    acc = round(result['accuracy'], 6)
    table = classification_report(output_list, target_list)
    loss = round(running_loss / len(data_loader.dataset), 6)

    return acc, loss

def parse_args():
    arg_parser = argparse.ArgumentParser(description="ResNet trained by CIFAR-10")
    
    arg_parser.add_argument("-k", "--model_width", nargs="?", type=int, default=1)
    arg_parser.add_argument("grad_steps", nargs="?", type=int, default=50000)
    arg_parser.add_argument("label_noise_rate", nargs="?", type=float, default=0)

    return arg_parser.parse_args()

if __name__ =='__main__':
    warnings.filterwarnings('ignore')
    # main()
    import time
    start = time.perf_counter()
    sub()
    print(time.perf_counter() - start)
