import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from newtools.dataset import DatasetTrain,DatasetVal
from lib.models import model_factory
from configs import cfg_factory
from lib.cityscapes_cv2 import get_data_loader
from tools.evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
from tqdm import tqdm, trange
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

import time

if __name__ == "__main__":
    # NOTE! NOTE! change this to not overwrite all log data when you train the model:
    # network = DeepLabV3(model_id=1, project_dir="E:/master/master1/RSISS/deeplabv3/deeplabv3").cuda()
    # x = Variable(torch.randn(2,3,256,256)).cuda() 
    # print(x.shape)
    # y = network(x)
    # print(y.shape)
    model_id = "1"

    num_epochs = 2
    batch_size = 1
    learning_rate = 0.001

    def parse_args():
        parse = argparse.ArgumentParser()
        parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
        parse.add_argument('--port', dest='port', type=int, default=44554,)
        parse.add_argument('--model', dest='model', type=str, default='bisenet_customer',)
        parse.add_argument('--finetune-from', type=str, default=None,)
        return parse.parse_args()

    args = parse_args()
    cfg = cfg_factory[args.model]
    network = model_factory[cfg.model_type](8)
    network.cuda()
    # network.load_state_dict(torch.load("training_logs/checkpoint/model_1_epoch_12.pth"))
    # network.load_state_dict(torch.load("training_logs/model_1/checkpoints/model_1_epoch_9.pth"))

    train_dataset = DatasetTrain(uavid_data_path="D:/BaiduNetdiskDownload/uavid/uavid_v1.5_official_release_split/UAVidDataset",
                                uavid_meta_path="D:/BaiduNetdiskDownload/uavid/uavid_v1.5_official_release_split/UAVidDataset",mypathi=r"J:\Desktop\bad\images",mypathl=r"J:\Desktop\bad\masks")
    val_dataset = DatasetVal(uavid_data_path="D:/BaiduNetdiskDownload/uavid/uavid_v1.5_official_release_split/UAVidDataset",
                            uavid_meta_path="D:/BaiduNetdiskDownload/uavid/uavid_v1.5_official_release_split/UAVidDataset",mypathi=r"J:\Desktop\bad\images",mypathl=r"J:\Desktop\bad\masks")

    num_train_batches = int(len(train_dataset)/batch_size)
    num_val_batches = int(len(val_dataset)/batch_size)
    print ("num_train_batches:", num_train_batches)
    print ("num_val_batches:", num_val_batches)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=1,drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size, shuffle=False,
                                            num_workers=1,drop_last=True)

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    # with open("D:/BaiduNetdiskDownload/cityscapes/class_weights.pkl", "rb") as file: # (needed for python3)
    #     class_weights = np.array(pickle.load(file))
    # class_weights = torch.from_numpy(class_weights)
    # class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    epoch_losses_train = []
    epoch_losses_val = []
    for epoch in range(num_epochs):
        print ("###########################")
        print ("######## NEW EPOCH ########")
        print ("###########################")
        print ("epoch: %d/%d" % (epoch+1, num_epochs))

        ############################################################################
        # train:
        ############################################################################
        network.train() # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (imgs, label_imgs) in tqdm(enumerate(train_loader)):
            #current_time = time.time()

            imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
            # print(imgs.shape)
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))
            # print(label_imgs.shape)
            outputs,*outputs_aux = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))
            # print(outputs)
            # print("shape of label_imgs: ",label_imgs.shape)
            # print("shape of outputs: ",outputs.shape)

            # compute the loss:
            loss = loss_fn(outputs, label_imgs)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            # optimization step:
            optimizer.zero_grad() # (reset gradients)
            loss.backward() # (compute gradients)
            optimizer.step() # (perform optimization step)

            #print (time.time() - current_time)

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)
        with open("%s/epoch_losses_train.pkl" % "training_logs", "wb") as file:
            pickle.dump(epoch_losses_train, file)
        print ("train loss: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_train, "k^")
        plt.plot(epoch_losses_train, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("train loss per epoch")
        plt.savefig("%s/epoch_losses_train.png" % "training_logs")
        plt.close(1)

        print ("####")

        ############################################################################
        # val:
        ############################################################################
        network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (imgs, label_imgs, img_ids) in tqdm(enumerate(val_loader)):
            with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
                imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
                label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))

                outputs,*outputs_aux = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

                # compute the loss:
                loss = loss_fn(outputs, label_imgs)
                loss_value = loss.data.cpu().numpy()
                batch_losses.append(loss_value)

        epoch_loss = np.mean(batch_losses)
        epoch_losses_val.append(epoch_loss)
        with open("%s/epoch_losses_val.pkl" % "training_logs", "wb") as file:
            pickle.dump(epoch_losses_val, file)
        print ("val loss: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_val, "k^")
        plt.plot(epoch_losses_val, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("val loss per epoch")
        plt.savefig("%s/epoch_losses_val.png" % "training_logs")
        plt.close(1)

        # save the model weights to disk:
        checkpoint_path = "training_logs/checkpoint" + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
        torch.save(network.state_dict(), checkpoint_path)
