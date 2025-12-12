
#imports
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

from torchsummary import summary
from tqdm import tqdm


#basic configuration stuff
torch.manual_seed(0)

#for Xander's local path
data_path = r"C:/Users/xjbar/Desktop/New folder/new plates"

train_dir = os.path.join(data_path, "train")
test_dir = os.path.join(data_path, "test")

class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)


#helper function to count images per class/split
def count_images(split_name: str, class_name: str) -> int:
    folder = os.path.join(data_path, split_name, class_name)
    return len(os.listdir(folder))


for label in class_names:
    train_count = count_images("train", label)
    test_count = count_images("test", label)
    print(f"{label:>12} | train: {train_count:3d}  test: {test_count:3d}")



#visualize some images using Pillow
def show_some_examples(num_classes_to_show: int = 4, num_images_per_class: int = 4):
    #display a few sample images from the training set
    for class_name in class_names[:num_classes_to_show]:
        class_folder = os.path.join(train_dir, class_name)
        image_files = sorted(os.listdir(class_folder))[:num_images_per_class]

        plt.figure(figsize=(num_images_per_class * 2.5, 3))
        for i, fname in enumerate(image_files):
            img_path = os.path.join(class_folder, fname)
            img = Image.open(img_path).convert("RGB")

            ax = plt.subplot(1, num_images_per_class, i + 1)
            ax.imshow(img)
            ax.set_title(class_name, fontsize=8)
            ax.axis("off")

        plt.suptitle(f"Examples from class: {class_name}")
        plt.tight_layout()
        plt.show()


#uncomment to show some examples
# show_some_examples()


#data transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


#data sets and data loaders
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=True,
    num_workers=0,  #can set to >0 but we leave at 0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    pin_memory=True,
    num_workers=0,
)

print("Number of training images:", len(train_dataset))
print("Number of test images:", len(test_dataset))


#device setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using device:", device)

if use_cuda:
    print("GPU Name:", torch.cuda.get_device_name(0))


#using ResNet18 + custom classifier
def build_model(num_classes: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    #freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    #replace final fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    #unfreeze the last block (layer4) and the new classifier head
    for name, param in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True

    return model


model = build_model(num_classes).to(device)
summary(model, input_size=(3, 224, 224))

#collect trainable parameters
params_to_update = [p for p in model.parameters() if p.requires_grad]
print(f"Number of trainable parameter tensors: {len(params_to_update)}")


#loss fucntion and optimizer
criterion = nn.CrossEntropyLoss()

#we use the Adam optimizer
optimizer = torch.optim.Adam(
    params_to_update,
    lr=3e-4,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1,
)


#training and evaluation functions
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    #train the model for one epoch and return loss and accuracy
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    #progress bar for visual feedback during training
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in progress_bar:
        #move data to device
        images = images.to(device)
        labels = labels.to(device)

        #reset gradients
        optimizer.zero_grad()
        #forward pass
        outputs = model(images)
        #compute loss
        loss = criterion(outputs, labels)

        #backpropagation
        loss.backward()
        optimizer.step()

        #accumulate loss
        running_loss += loss.item() * images.size(0)
        #get predicted classes
        preds = outputs.argmax(dim=1)
        #count correct predictions
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        #compute running metrics for progress bar
        current_loss = running_loss / total
        current_acc = 100.0 * correct / total
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    #compute epoch-level metrics
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
):
    #evaluate the model on validation or test data
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    #compute final evaluation metrics
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


#main training loop
EPOCHS = 30

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
}

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, device)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Train | loss: {train_loss:.4f}  acc: {train_acc:.2f}%\n"
        f"Val   | loss: {val_loss:.4f}  acc: {val_acc:.2f}%\n"
        f"LR    | {optimizer.state_dict()['param_groups'][0]['lr']:.6f}"
    )

    #save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_resnet18_plates.pth")
        print(f"--> New best val accuracy: {best_val_acc:.2f}%, model saved.")


# plot training curves
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history["train_loss"], label="Train loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss over epochs")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history["val_loss"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss over epochs")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history["train_acc"], label="Train acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy over epochs")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history["val_acc"], label="Val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy over epochs")

plt.tight_layout()
plt.show()