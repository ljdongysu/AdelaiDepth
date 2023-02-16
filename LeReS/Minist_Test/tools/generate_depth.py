import cv2
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm

from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt

def parse_args():
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')
    parser.add_argument('--load_ckpt', default='./res50.pth', help='Checkpoint path to load')
    parser.add_argument('--backbone', default='resnext101', help='Checkpoint path to load')
    parser.add_argument('--image_path', help='image path', required=True)
    parser.add_argument('--file_name', help='image list in file', required=True)

    args = parser.parse_args()
    return args

def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

def MkdirSimple(path):
    path_current = path
    suffix = os.path.splitext(os.path.split(path)[1])[1]

    if suffix != "":
        path_current = os.path.dirname(path)
        if path_current in ["", "./", ".\\"]:
            return
    if not os.path.exists(path_current):
        os.makedirs(path_current)

def get_file_name(file_name, image_path):
    with open(file_name, 'r') as f:
        file_lists = f.readlines()
    image_list = []
    depth_list = []
    for image_name in file_lists:
        image_full_path = os.path.join(image_path, image_name)
        image_dest_path = image_full_path.replace("REMAP", "DEPTH/AdelaiDepth")
        MkdirSimple(image_dest_path)
        image_list.append(image_full_path)
        depth_list.append(image_dest_path)

    return image_list, depth_list


def generate_depth(imgs_list, depth_list, depth_model):
    for i, v in enumerate(tqdm(imgs_list)):
        print('processing (%04d)-th image... %s' % (i, v))
        rgb = cv2.imread(v.split()[0])
        rgb_c = rgb[:, :, ::-1].copy()
        A_resize = cv2.resize(rgb_c, (448, 448))

        img_torch = scale_torch(A_resize)[None, :, :, :]
        pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
        pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

        write_name = depth_list[i].replace(".jpg",".png")
        write_name = write_name.split()[0]
        cv2.imwrite(write_name, (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))


if __name__ == '__main__':

    args = parse_args()

    # create depth model
    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()

    # load checkpoint
    load_ckpt(args, depth_model, None, None)
    depth_model.cuda()
    img_list, depth_list = get_file_name(args.file_name, args.image_path)

    generate_depth(img_list, depth_list, depth_model)

