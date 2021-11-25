import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms;
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
from model import resnet18k 

import argparse
import random
import matplotlib.pyplot as plt
import csv 
import warnings
    
def main():
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
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
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
    trloss = []
    teloss = []
    x2.append(0)
    train_data_save = []
    train_data_save.append(["step", "error", "loss"])
    train_acc, train_loss = test(model, device, test_loader, criterion)
    #test()を使ってtrainデータのerrorとlossの初期値を取得
    train_data_save.append([0, 1.0 - train_acc, train_loss])
    trerr.append(1.0 - train_acc)
    trloss.append(train_loss)

    test_data_save = []
    test_data_save.append(["step", "error", "loss"])
    test_acc, test_loss = test(model, device, test_loader, criterion)
    #test()を使ってtestデータのerrorとlossの初期値を取得
    test_data_save.append([0, 1.0 - test_acc, test_loss])
    teerr.append(1.0 - test_acc)
    teloss.append(test_loss)

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

            train_acc, train_loss = calc_score(target_list, output_list, running_loss, train_loader)
            if count % 1000 == 0 and count != 0:
                stdout_temp = 'step: {:>3}/{:<3}, train acc:{:<8}, train loss: {:<8}'
                print(stdout_temp.format(count, grad_steps, train_acc, train_loss))
            trerr.append(1.0 - train_acc)
            trloss.append(loss.item())
            count += 1
            scheduler.step()
            train_data_save.append([count, 1.0 - train_acc, train_loss])
        test_acc, test_loss = test(model, device, test_loader, criterion)
        stdout_temp = 'step: {:>3}/{:<3}, test acc:{:<8}, test loss: {:<8}'
        print(stdout_temp.format(count, grad_steps, test_acc, test_loss))
        teerr.append(1.0 - test_acc)
        teloss.append(test_loss)
        test_data_save.append([count, 1.0 - train_acc, train_loss])
        x2.append(count)
    
    with open('./csv/resnet18_' + str(args.model_width) + '-cifar10-train.csv','w') as file:
        writer = csv.writer(file)
        writer.writerows(train_data_save)

    with open('./csv/resnet18_' + str(args.model_width) + '-cifar10-test.csv','w') as file:
        writer = csv.writer(file)
        writer.writerows(test_data_save)
    
    #errorのグラフ化
    plt.title("ResNet18*" + str(args.model_width) + " trained by Cifar10")
    plt.xlim(0, grad_steps * 1.2)
    plt.ylim(0, 1)
    plt.xlabel("Steps")
    plt.ylabel("Erorr")
    plt.plot(x1, trerr, label = 'train', linewidth=0.5)
    plt.plot(x2, teerr, marker='.', label='test')
    plt.legend(loc='upper right')
    plt.savefig("./output/ResNet18_" + str(args.model_width) + "TrainedByCifar10.png")

    plt.close()

    #lossのグラフ化
    plt.title("ResNet18*" + str(args.model_width) + " trained by Cifar10")
    plt.xlim(0, grad_steps * 1.2)
    plt.ylim(0, max(max(trloss) * 1.2, max(teloss) * 1.2))
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.plot(x1, trloss, label = 'train', linewidth=0.5)
    plt.plot(x2, teloss, marker='.', label='test')
    plt.legend(loc='upper right')
    plt.savefig("./output/loss-ResNet18_" + str(args.model_width) + "TrainedByCifar10.png")

def sub():
    args = parse_args()
    #epoch数指定
    epoch = args.grad_steps
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
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    x1 = []
    x2 = range(epoch + 1)
    trerr = []
    teerr = []
    trloss = []
    teloss = []
    train_data_save = []

    x1.append(0)
    train_data_save.append(["epoch", "error", "loss"])
    train_acc, train_loss = test(model, device, test_loader, criterion)
    #test()を使ってtrainデータのerrorとlossの初期値を取得
    train_data_save.append([0, 1.0 - train_acc, train_loss])
    trerr.append(1.0 - train_acc)
    trloss.append(train_loss)

    test_data_save = []
    test_data_save.append(["epoch", "error", "loss"])
    test_acc, test_loss = test(model, device, test_loader, criterion)
    #test()を使ってtestデータのerrorとlossの初期値を取得
    test_data_save.append([0, 1.0 - test_acc, test_loss])
    teerr.append(1.0 - test_acc)
    teloss.append(test_loss)

    for epoch in range(epoch):
        # Train and test a model.
        model.train()
        #trainの各数値はbatchごとに出す
        #calc_scoreの返却値のlossはbatch数で割っているので無視,loss.item()を用いる
        for batich_idx, (inputs, targets) in enumerate(train_loader):
            output_list = []
            target_list = []
            running_loss = 0.0
            xpoint = 0.0 + epoch + (float(batich_idx + 1) / len(train_loader))
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output_list += [int(o.argmax()) for o in outputs]
            target_list += [int(t) for t in targets]
            running_loss += loss.item()

            train_acc, train_loss = calc_score(target_list, output_list, running_loss, train_loader)
            x1.append(xpoint)
            trerr.append(1.0 - train_acc)
            trloss.append(loss.item())
            train_data_save.append([xpoint, 1.0 - train_acc, loss.item()])
            # if batich_idx % 100 == 0 and batich_idx != 0:
            #     stdout_temp = 'batch: {:>3}/{:<3}, train acc:{:<8}, train loss: {:<8}'
            #     print(stdout_temp.format(batich_idx, len(train_loader), train_acc, loss.item()))
            
        test_acc, test_loss = test(model, device, test_loader, criterion)

        teerr.append(1.0 - test_acc)
        teloss.append(test_loss)
        test_data_save.append([epoch + 1, 1.0 - test_acc, test_loss])
        # Output score.
        stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, test acc: {:<8}, test loss: {:<8}'
        print(stdout_temp.format(epoch+1, train_acc, loss.item(), test_acc, test_loss))
    
    torch.save(model.state_dict(), './model_weight/resnet18*' + str(args.model_width) + '-cifar10-train.csv')

    with open('./csv/resnet18*' + str(args.model_width) + '-cifar10-train.csv','w') as file:
        writer = csv.writer(file)
        writer.writerows(train_data_save)

    with open('./csv/resnet18*' + str(args.model_width) + '-cifar10-test.csv','w') as file:
        writer = csv.writer(file)
        writer.writerows(test_data_save)
    
    #errorのグラフ化
    plt.title("ResNet18*" + str(args.model_width) + " trained by Cifar10")
    plt.xlim(0, epoch * 1.2)
    plt.ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Erorr")
    plt.plot(x1, trerr, label='train', linewidth=0.5)
    plt.plot(x2, teerr, label='test')
    plt.legend(loc='upper right')
    plt.savefig("./output/ResNet18*" + str(args.model_width) + "TrainedByCifar10.png")

    plt.close()

    #lossのグラフ化
    plt.title("ResNet18*" + str(args.model_width) + " trained by Cifar10")
    plt.xlim(0, epoch * 1.2)
    plt.ylim(0, max(max(trloss) * 1.2, max(teloss) * 1.2))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(x1, trloss, label='train', linewidth=0.5)
    plt.plot(x2, teloss, label='test')
    plt.legend(loc='upper right')
    plt.savefig("./output/loss-ResNet18*" + str(args.model_width) + "TrainedByCifar10.png")

    

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

        train_acc, train_loss = calc_score(target_list, output_list, running_loss, train_loader)
        if batich_idx % 100 == 0 and batich_idx != 0:
            stdout_temp = 'batch: {:>3}/{:<3}, train acc:{:<8}, train loss: {:<8}'
            print(stdout_temp.format(batich_idx, len(train_loader), train_acc, train_loss))
    train_acc, train_loss = calc_score(target_list, output_list, running_loss, train_loader)


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
		
	test_acc, test_loss = calc_score(target_list, output_list, running_loss, test_loader)

	return test_acc, test_loss
def calc_score(true_list, predict_list, running_loss, data_loader):
    # import pdb;pdb.set_trace()
    # result = classification_report(true_list, predict_list, output_dict=True)
    # acc = round(result['accuracy'], 6)
    acc = accuracy_score(true_list, predict_list)
    loss = round(running_loss / len(data_loader), 6)

    return acc, loss

def parse_args():
    arg_parser = argparse.ArgumentParser(description="ResNet trained by CIFAR-10")
    
    arg_parser.add_argument("-k", "--model_width", nargs="?", type=int, default=1)
    arg_parser.add_argument("grad_steps", nargs="?", type=int, default=50000)
    arg_parser.add_argument("label_noise_rate", nargs="?", type=float, default=0.0)

    return arg_parser.parse_args()

if __name__ =='__main__':
    warnings.filterwarnings('ignore')
    import time
    start = time.perf_counter()
    sub()
    print(time.perf_counter() - start)
