import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import efficientnetv2_s as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform_png = transforms.Compose(
        [transforms.Resize((384, 384)),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
         ])
    data_transform_jpg = transforms.Compose(
        [transforms.Resize((384, 384)),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    img_path = "/Users/maojietang/Downloads/Test_1(100%)/train/bishop_resized/125_6cc58fb4d242103136854c4a895ca3ea.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    channel = len(img.split())
    if channel not in [3, 4]:
        return print('Image Channel is not Match, the channel is {}'.format(channel))
    # [N, C, H, W]
    plt.imshow(img)
    if channel == 4:
        img = data_transform_png(img)
        img = img[0:3, :, :]
    else:
        img = data_transform_jpg(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_label.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=5).to(device)
    # load model weights
    model_weight_path = "/Users/maojietang/Downloads/EfficientV2_S.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
