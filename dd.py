import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms;
from sklearn.metrics import classification_report

import warnings

def main():
    warnings.filterwarnings('ignore')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip,
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
    model = torchvision.models.resnet18(pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(10):
        train_acc, train_loss = train(model, device, train_loader, criterion, optimizer)
        test_acc, test_loss = test(model, device, test_loader, criterion)
        stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, test acc: {:<8}, test loss: {:<8}'
        print(stdout_temp.format(epoch+1, train_acc, train_loss, test_acc, test_loss))
        print('')

def train (model, device, train_loader, criterion, optimizer):
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
    acc = round(result['weighted avg']['f1-score'], 6)
    loss = round(running_loss / len(data_loader.dataset), 6)

    return acc, loss

    
if __name__ =='__main__':
    main()
