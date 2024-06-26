# Michael Davies
# mdavies1@vt.edu
#
# code citations:
#  https://nextjournal.com/gkoehler/pytorch-mnist
#  https://www.kaggle.com/code/amirhosseinzinati/animal10-with-resnet50-in-pytorch
import timeit

# imports
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# static values
root = "images"

# turn baseline linear model on/off (False means it does not run)
linear = False

# parameters
n_epochs = 25
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 99
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

##-----------------------------------------------------##
## Load Animals-10 dataset from local directory images ##
##-----------------------------------------------------##

# create transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# create dataset
Dataset = datasets.ImageFolder(root=root, transform=transform)
# print basic data information
num_classes = len(Dataset.classes)
print(f"Number of classes: {num_classes}")
print(*Dataset.classes, sep=", ")
num_images = len(Dataset)
print(f"Number of samples: {num_images}")

# split dataset for train and test 80/20
train_size = int(0.8 * len(Dataset))
test_size = len(Dataset) - train_size
train_dataset, test_dataset = random_split(Dataset, [train_size, test_size])

# create dataloaders for both datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size_train)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test)

# visualize examples
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)

fig = plt.figure()
for i in range(15):
    plt.subplot(5, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0])
    plt.title("Ground Truth: {}".format(Dataset.classes[example_targets[i]]))
    plt.xticks([])
    plt.yticks([])
plt.show()

##------------------------------------------##
## Build a Linear Baseline Model ##
##------------------------------------------##

if linear:

    # build the linear model
    class LinearNet(nn.Module):
        def __init__(self):
            super(LinearNet, self).__init__()
            self.fc1 = nn.Linear(in_features=150528, out_features=6)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            return F.log_softmax(x, dim=1)


    # initialize the network and optimizer
    linear_network = LinearNet()
    linear_optimizer = optim.SGD(linear_network.parameters(), lr=learning_rate, momentum=momentum)

    # train the model
    linear_train_losses = []
    linear_train_counter = []
    linear_test_losses = []
    linear_test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


    def linear_train(epoch):
        linear_network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            linear_optimizer.zero_grad()
            output = linear_network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            linear_optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.item()))
                linear_train_losses.append(loss.item())
                linear_train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
                torch.save(linear_network.state_dict(), 'linear_model.pth')
                torch.save(linear_optimizer.state_dict(), 'linear_optimizer.pth')


    def linear_test():
        linear_network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = linear_network(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        linear_test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


    linear_test()
    for epoch in range(1, n_epochs + 1):
        linear_train(epoch)
        linear_test()

    fig = plt.figure()
    plt.plot(linear_train_counter, linear_train_losses, color='blue')
    plt.scatter(linear_test_counter, linear_test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.title('Linear Model')
    plt.show()


# exit("ending early for testing")

##---------------------##
## Build the CNN Model ##
##---------------------##

# start train/test timer
timer_start = timeit.default_timer()


# build the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)
        self.fc1 = nn.Linear(in_features=28800, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=6)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# initialize the network and optimizer
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# train the model
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), 'model.pth')
            torch.save(optimizer.state_dict(), 'optimizer.pth')


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

# end train/test timer
timer_end = timeit.default_timer()
print('CNN Model time to build, train, and test: ', timer_end - timer_start, ' seconds')

# view results
fig = plt.figure()
plt.plot(train_counter, train_losses, color='green')
plt.scatter(test_counter, test_losses, color='orange')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.title('CNN Model')
plt.show()

# view example results
with torch.no_grad():
    output = network(example_data)

fig = plt.figure()
for i in range(15):
    plt.subplot(5, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0])
    plt.title("Prediction: {}".format(Dataset.classes[output.data.max(1, keepdim=True)[1][i].item()]))
    plt.xticks([])
    plt.yticks([])
plt.show()
