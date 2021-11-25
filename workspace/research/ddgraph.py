import os
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd
x = []
y = []
y2 = []
for i in range(64):
    if (os.path.isfile('./csv/resnet18*' + str(i + 1) + '-cifar10-test.csv')):
        x.append(i + 1)
        data = pd.read_csv('./csv/resnet18*' + str(i + 1) + '-cifar10-test.csv')
        err = data.values.tolist()[-1][1]
        y.append(1.0 - (1.0 - err) * (1.0 - 0.15) + err * 0.15 / 9.0)
        # y2.append(err)

plt.title("Double Descent")
plt.xlim(0, 65)
plt.ylim(0.0, max(y))
plt.xlabel("ResNet18 width parameter")
plt.ylabel("Test Error")
plt.plot(x, y, label='test')
# plt.plot(x, y2, label='nashi')
plt.legend(loc='upper right')
plt.savefig("./DoubleDescent.png")

        
        
 