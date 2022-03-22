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
    # image's file
    img_root = "/Users/maojietang/Downloads/Test_1(100%)/val/"
    Standard = 'bishop_resized'
    imgs_root = os.path.join(img_root, Standard)
    Fail_file = []
    Error_file = []

    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # image's format
    format = ['jpeg', 'jpg', 'png']
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.split('.')[-1] in format]

    # read class_indict
    json_path = './class_label.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=5).to(device)

    # load model weights
    weights_path = "/Users/maojietang/Downloads/EfficientV2_S.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    batch_size = 8  # batch size
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                channel = len(img.split())
                if len(img.split()) not in [3, 4]:
                    Fail_file.append(img_path)
                    continue
                elif channel == 4:
                    img = data_transform_png(img)
                    img = img[0:3, :, :]
                else:
                    img = data_transform_jpg(img)

                img_list.append(img)

            # batch img
            # batch all images in img_list
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))
                if class_indict[str(cla.numpy())] != Standard:
                    Error_file.append(img_path_list[ids * batch_size + idx])

        # show fail and error files
        print('*'*40)
        print(Fail_file)
        print('*'*40)
        print(Error_file)


if __name__ == '__main__':
    main()
