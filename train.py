from torchvision.transforms.transforms import RandomVerticalFlip
import os
import json
import sys

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision.models as models



# Set Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using {} for this project'.format(device))

# Image_Loader and Label
# path_root = '/Users/maojietang/Downloads/Image_Segmentation-main/deep-learning-for-image-processing-master'
path_root = '/content/drive/MyDrive'
image_path = os.path.join(path_root, 'data_642')
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop((300, 300)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "val": transforms.Compose([transforms.Resize((384, 384)),
                                transforms.CenterCrop((384, 384)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

# Get train_set
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                      transform=data_transform['train'])
train_num = len(train_dataset)

# Get label
category_to_label = train_dataset.class_to_idx
label_to_category = dict((val, key) for key, val in category_to_label.items())

# Write to json
json_label = json.dumps(label_to_category, indent=4)
with open('class_label.json', 'w') as json_file:
    json_file.write(json_label)

# Load train_loader / val_loader
batch_size = 16
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=nw)
batch_num = len(train_loader)

val_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                                    transform=data_transform['val'])
val_num = len(val_dataset)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=nw)

# Model Initial
# Transfer Learning
model = efficientnetv2_s(num_classes=1000)
model_path = torch.load('/content/drive/MyDrive/pre_efficientnetv2-s.pth')
model.load_state_dict(model_path)
model.head.classifier = nn.Sequential(
    nn.Linear(in_features=1280, out_features=1000, bias=True),
    nn.Linear(in_features=1000, out_features=5, bias = True)
)
for m in model.parameters():
  m.grad_require = False

model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00028)

# Train
epochs = 100
best_acc = 0.0
save_path = './EfficientV2_S16.pth'
test_loss_curve = []
train_loss_curve = []
for epoch in range(epochs):
    model.train()
    train_bar = tqdm(train_loader, file=sys.stdout)
    total_loss = 0.0
    for step, data in enumerate(train_bar):
        optimizer.zero_grad()
        img, label = data
        predict = model(img.to(device))
        loss = loss_function(predict, label.to(device))
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    print('Epoch[{}/{}] train_loss: {}'.format(epoch + 1,
                                                  epochs,
                                                  total_loss))
    train_loss_curve.append(total_loss/batch_size)

    model.eval()
    val_bar = tqdm(val_loader, file=sys.stdout)
    acc = 0.0
    with torch.no_grad():
        for data in val_bar:
            val_img, val_label = data
            predict = model(val_img.to(device))
            output = torch.max(predict, dim=1)[1]
            acc += torch.eq(output, val_label.to(device)).sum().item()
    accuracy = acc / val_num
    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                epochs)
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
          (epoch + 1, total_loss / batch_num, accuracy))
    test_loss_curve.append(accuracy)

    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), save_path)
print('Finish Training!')
plt.plot(range(epochs), test_loss_curve, label='test')
plt.plot(range(epochs), train_loss_curve, label='train')
plt.legend()
plt.show()
print('Best Acc:{}'.format(best_acc))
print('Transfer Learing: Add New and Train Last')


