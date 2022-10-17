
from cgi import test
import sys
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import os
import datetime
import lib.transform_cv2 as T
from lib.models import model_factory
from configs import cfg_factory

torch.set_grad_enabled(False)
np.random.seed(123)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='/root/YangZR/datasets/mix/test',)
args = parse.parse_args()
cfg = cfg_factory[args.model]


palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](19)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
net.cuda()

# prepare data
# 这是要改的，数据集不同，均值方差也不同
#小数据集bad
# to_tensor = T.ToTensor(
#      mean=(0.261070850317029, 0.2615512058423913, 0.26178125), # city, rgb
#     std=(0.2511606324046974, 0.250846449744934, 0.2507283462923785),
# )

#大数据集mix
to_tensor = T.ToTensor(
     mean=(0.2569073932926829, 0.2573543889735772, 0.2575796836555948), # city, rgb
    std=(0.24859957653845138, 0.2482726899155714, 0.2481900670474492),
)
'''-------------------------原版
im = cv2.imread(args.img_path)[:, :, ::-1]
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

# inference
out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
pred = palette[out]
cv2.imwrite('./res.jpg', pred)
------------------------------'''
test_imgs=os.listdir(args.img_path)
for testimg in test_imgs:
    test_img_path=os.path.join(args.img_path,testimg)
    im = cv2.imread(test_img_path)[:, :, ::-1]
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
    # inference
    out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
    pred = palette[out]
    res_img_path = os.path.join("/root/YangZR/BiSeNetOFF/res/out",testimg)
    print(testimg)
    cv2.imwrite(res_img_path, pred)