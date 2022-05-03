from model import efficientnetv2_s
import torch
import torch.nn as nn
import torch.optim as optim
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate
import os
from torchvision import datasets, transforms
import json

# Set Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using {} for this project'.format(device))

# Initialize
# Model path
weight_path = '/Users/maojietang/Downloads/pre_efficientnetv2-s.pth'
# Data path'
path_root = "/Users/maojietang/Downloads/Chess_Piece_Data"
image_path = os.path.join(path_root, 'Test_1(100%)')
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
nw = 0#min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
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

# Get Pretrain weights
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def net_train(net, train_loader, parameters, dtype, device):
  net.to(dtype=dtype, device=device)


  # Define loss and optimizer
  criterion = nn.CrossEntropyLoss()
  # creterion = LabelSmoothingLoss(classes=5, smoothing = parameters.get("smoothing", 0.9))
  optimizer = optim.Adam(net.parameters(), # or any optimizer you prefer
                        lr=parameters.get("lr", 0.001) # 0.001 is used if no lr is specified
  )


  scheduler = optim.lr_scheduler.StepLR(
      optimizer,
      step_size=int(parameters.get("step_size", 30)),
      gamma=parameters.get("gamma", 1.0),  # default is no learning rate decay
  )


  num_epochs = parameters.get("num_epochs", 20) # Play around with epoch number
  # Train Network
  for _ in range(num_epochs):
      for inputs, labels in train_loader:
          # move data to proper dtype and device
          inputs = inputs.to(dtype=dtype, device=device)
          labels = labels.to(device=device)


          # zero the parameter gradients
          optimizer.zero_grad()


          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          scheduler.step()
  return net


def init_net(parameterization):
    global weight_path
    model = efficientnetv2_s(num_classes=1000)
    model_path = torch.load(weight_path)
    model.load_state_dict(model_path)

    for m in model.parameters():
        m.requires_grad = False

    in_feature = model.head.classifier.in_features
    model.head.classifier = nn.Linear(in_feature, 5)

    return model  # return untrained model


def train_evaluate(parameterization):
    # constructing a new training data loader allows us to tune the batch size
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=parameterization.get("batchsize", 32),
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)

    # Get neural net
    untrained_net = init_net(parameterization)

    # train
    trained_net = net_train(net=untrained_net, train_loader=train_loader,
                            parameters=parameterization, dtype=dtype, device=device)

    # return the accuracy of the model as it was trained in this run
    return evaluate(
        net=trained_net,
        data_loader=val_loader,
        dtype=dtype,
        device=device,
    )

def main():
    dtype = torch.float
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # {'lr': 0.001, 'gamma': 1.0, 'stepsize': 36}

    best_parameters, values, experiment, model = optimize(
        # Set parameters for tuning
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-5, 1e-3]},
            {"name": "gamma", "type": "range", "bounds": [0.95, 1.0]},
            {"name": "stepsize", "type": "range", "bounds": [20, 40]},
            # {"name": "smoothing", "type": "range", "bounds": [0.1, 1.0]}
        ],

        evaluation_function=train_evaluate,
        objective_name='accuracy',
    )

    print(best_parameters)
    means, covariances = values
    print(means)
    print(covariances)