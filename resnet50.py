#imports
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

from torchsummary import summary
from tqdm import tqdm

from torch_lr_finder import LRFinder  # pip install torch-lr-finder

#basic configuration stuff
torch.manual_seed(0)

#for Xander's local path
data_path = r"C:/Users/xjbar/Desktop/New folder/new plates"

#list classes (subfolders under train/)
class_names = sorted(os.listdir(os.path.join(data_path, "train")))
print("Classes:", class_names)
print("Number of classes:", len(class_names))

#helper function to count images per class/split
def get_count(set_type, state):
    folder = os.path.join(data_path, set_type, state)
    return len(os.listdir(folder))


for each_class in class_names:
    for set_type in ["train", "test"]:
        print(
            f"Number of {set_type} samples in {each_class}: "
            f"{get_count(set_type, each_class)}"
        )

#visualize some images using Pillow
train_dir_list = os.listdir(os.path.join(data_path, "train"))

#display a few sample images from the training set
for each in train_dir_list[:5]:
    current_folder = os.path.join(data_path, "train", each)
    plt.figure(figsize=(10, 2))

    for i, file in enumerate(os.listdir(current_folder)[:5]):
        fullpath = os.path.join(current_folder, file)
        img = Image.open(fullpath).convert("RGB")

        ax = plt.subplot(1, 5, i + 1)
        ax.set_title(each, fontsize=8)
        ax.axis("off")
        ax.imshow(img)

    plt.tight_layout()
    plt.show()

#data transforms
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.GaussianBlur(5, sigma=0.60),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing()
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#data sets and data loaders
train_data = datasets.ImageFolder(
    os.path.join(data_path, "train"),
    transform=train_transform
)
test_data = datasets.ImageFolder(
    os.path.join(data_path, "test"),
    transform=test_transform
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, pin_memory=True)

print("Number of train images:", len(train_data))
print("Number of test images:", len(test_data))

#device setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using device:", device)

if use_cuda:
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Memory Allocated:", torch.cuda.memory_allocated(0))
    print("Memory Reserved:", torch.cuda.memory_reserved(0))

#using ResNet50 + custom classifier
model = resnet50(weights=ResNet50_Weights.DEFAULT)

#show base model summary
summary(model, input_size=(3, 224, 224))

def set_parameter_requires_grad(m, feature_extracting=True):
    if feature_extracting:
        for name, param in m.named_parameters():
            param.requires_grad = False

#freeze all layers initially
set_parameter_requires_grad(model, feature_extracting=True)

#replace final fully connected layer
model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, len(class_names)),
)

#unfreeze the last block (layer4) and the new classifier head
for name, param in model.named_parameters():
    if "layer4" in name:
        param.requires_grad = True

#collect parameters to update
params_to_update = []
print("Params to be updated:")
for name, param in model.named_parameters():
    if param.requires_grad:
        params_to_update.append(param)
        print("\t", name)

model = model.to(device)
print(model.fc)

#loss and tracking lists
criterion = nn.CrossEntropyLoss()

train_losses = []
test_losses = []
train_acc = []
test_acc = []

#train and test functions
def train_one_epoch(model, device, train_loader, optimizer, epoch):
    #set model to training mode
    model.train()
    #create progress bar
    pbar = tqdm(train_loader)
    #counters for accuracy
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        #move data to device
        data, target = data.to(device), target.to(device)

        #reset gradients
        optimizer.zero_grad()
        #forward pass
        outputs = model(data)
        #compute loss
        loss = criterion(outputs, target)

        #store training loss
        train_losses.append(loss.detach().cpu())

        #backpropagation
        loss.backward()
        optimizer.step()

        #get predictions
        preds = outputs.argmax(dim=1, keepdim=True)
        #count correct predictions
        correct += preds.eq(target.view_as(preds)).sum().item()
        processed += len(data)

        #update progress bar
        pbar.set_description(
            f"Epoch {epoch} Loss={loss.item():.4f} "
            f"Batch={batch_idx} Acc={100*correct/processed:0.2f}%"
        )

    #compute and store epoch accuracy
    epoch_acc = 100 * correct / processed
    train_acc.append(epoch_acc)

def test_model(model, device, test_loader):
    #set model to evaluation mode
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        #disable gradient computation
        for data, target in test_loader:
            #move data to device
            data, target = data.to(device), target.to(device)
            #forward pass
            outputs = model(data)
            #compute loss
            loss = criterion(outputs, target)

            #accumulate test loss
            test_loss += loss.detach().cpu().item()
            #compute predictions
            preds = outputs.argmax(dim=1, keepdim=True)
            #count correct predictions
            correct += preds.eq(target.view_as(preds)).sum().item()

    #compute average loss
    test_loss /= len(test_loader)
    #compute accuracy
    accuracy = 100.0 * correct / len(test_loader.dataset)
    #store metrics
    test_losses.append(test_loss)
    test_acc.append(accuracy)

    #print evaluation summary
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return accuracy


#learning rate finder
lr_device = "cuda" if use_cuda else "cpu"
optimizer_tmp = torch.optim.SGD(params_to_update, lr=1e-7, weight_decay=1e-2)
lr_finder = LRFinder(model, optimizer_tmp, criterion, device=lr_device)

lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
lr_finder.plot()
lr_finder.reset()


#final training setup
optimizer = torch.optim.Adam(params_to_update, lr=0.0001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[10, 15], gamma=0.1
)

EPOCHS = 30

for epoch in range(1, EPOCHS + 1):
    print("EPOCH:", epoch)
    train_one_epoch(model, device, train_loader, optimizer, epoch)
    scheduler.step()
    print("Current Learning Rate:", optimizer.state_dict()["param_groups"][0]["lr"])
    test_accu = test_model(model, device, test_loader)


#plot training curves
train_losses1 = [float(x) for x in train_losses]
test_losses1 = [float(x) for x in test_losses]

fig, axs = plt.subplots(2, 2, figsize=(16, 10))

axs[0, 0].plot(train_losses1)
axs[0, 0].set_title("Training Loss")

axs[1, 0].plot(train_acc)
axs[1, 0].set_title("Training Accuracy")

axs[0, 1].plot(test_losses1)
axs[0, 1].set_title("Test Loss")

axs[1, 1].plot(test_acc)
axs[1, 1].set_title("Test Accuracy")

plt.tight_layout()
plt.show()
