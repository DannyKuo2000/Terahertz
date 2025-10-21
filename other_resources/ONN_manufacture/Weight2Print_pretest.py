# 這是一段 從訓練好的光學神經網路 (D2NN / ONN) 模型中取出各層參數，並轉換成實際可製作的三維光學結構資料 (3D voxel model) 的程式。
# 它的目的不是訓練模型，而是 根據模型的 phase pattern 產生對應的 3D 幾何結構 (.npy 檔案)，讓你之後能做模擬或實體實作。
import os
import csv
import random
import pathlib
import argparse
from tokenize import Double
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pytorch_ssim
import math
import D2NN_ONN_DNN_pretest_absorption_v0 # import your model here

i = (-1)**0.5

def find_surface_points(data):
    # Ensure the data is treated as a boolean array
    data = data.astype(bool)
    
    # Create an empty boolean array of the same shape as the input
    surface = np.zeros_like(data, dtype=bool)

    # Check each point in the array, avoiding the edges to prevent index errors
    for i in range(1, data.shape[0] - 1):
        for j in range(1, data.shape[1] - 1):
            for k in range(1, data.shape[2] - 1):
                # A point is considered a surface point if it is True and at least one of its neighbors is False
                if data[i, j, k] and not data[i-1:i+2, j-1:j+2, k-1:k+2].all():
                    surface[i, j, k] = True

    # Handle edges to ensure surface definition includes the boundary points
    # First layer and last layer along each axis
    surface[0, :, :], surface[-1, :, :] = data[0, :, :], data[-1, :, :]
    surface[:, 0, :], surface[:, -1, :] = data[:, 0, :], data[:, -1, :]
    surface[:, :, 0], surface[:, :, -1] = data[:, :, 0], data[:, :, -1]

    return surface

def main(args):

    vol_size = 0.000025  # voxel厚度 (單位 m)
    #vol_size = 0.000005  
    x_size = 0.00040   # 印製最小可操控的寬度 (m)
    H_thickness = 0.003  # 整個材料層的厚度 (m)
    H_number = int(H_thickness/vol_size)  # 厚度最大的可調整範圍
    dx = 0.00075  # pixel spacing
    x_number = int(dx/x_size)  # 一單位製造寬度對應到幾個weighting參數
    n = 1.70  # 折射率
    #n = 3.42
    dn = n - 1  # 折射率差
    c = 2.998e8  # 光速
    lam = c / 0.2004e12  # 波長 (0.2 THz)
    k = 2 * math.pi / lam  # 波數

    nn = 0

    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    model = D2NN_ONN_DNN_pretest_absorption_v0.Net()  # load your model
    model.cuda()  # move to cuda
    model.load_state_dict(torch.load(args.model_save_path + str(args.start_epoch) + args.model_name))  # load your weight

    for name, param in model.named_parameters():

        #print(torch.exp(i*param))
        #print("+++++++++++++++++++++++")
        
        size = param.size()
        sig = torch.nn.Sigmoid()
        param = sig(param)  # 用 Sigmoid 將參數壓到 (0, 1), why not linear transform ?????
        param = param * 2 * math.pi  # 再乘上2π，把它視為「相位偏移」
        
        print("+++++++++++++++++++++++++")
        print(param)
        print(param.shape)
        print(name)
        print("+++++++++++++++++++++++++")
        
        #param = torch.sin(param)*math.pi
        #param = param + math.pi
        

        nn = nn + 1


        #param = torch.sin(param)+1
        #param/=2
        #param = param*lam/dn
        #print(param)

        

        #nn = nn +1

        #print(torch.exp(i*(param)))


        cube = np.zeros((int(size[0] * x_number), int(size[1] * x_number), int(H_number)))  # 一片ONN的長寬高
        d_max = lam / dn  # not sure what is this?
        print("min ", torch.min(param), "max ", torch.min(param))
        min_h = 0.00000  # 最小限制厚度
        max_h = 1.00000  # 最大限制厚度


        for x in range(size[0]):
            for y in range(size[1]):

                h = param[x, y] / k / dn  # 相位轉成厚度, radian / (radian/length) / (折射率) = length, 好像可以不用乘2pi再除2pi
                
                #print("333333333333333333333333333333",param[x,y],"333333333333333333333333333333")
                #h = param[x,y]*d_max

                if (float(h) > max_h):
                    max_h = float(h)
                if (float(h) < min_h):
                    min_h = float(h)


                h = int(torch.round(h / vol_size))  # quantize到整數個voxel
                

                #print(h)

                plane = np.ones((x_number, x_number, h))  # 一個單位柱子的大小長相
                cube[0 + x_number * x : 0 + x_number * (x + 1), 0 + x_number * y : 0 + x_number * (y + 1), : h] = plane  # 把需要製造的單位柱子填成1，否則是0

        #np.save("param_pixel_{}.npy".format(nn), cube)
        print("22222222222222222222222222222222222222222222222")
        print("min_h ", min_h, "max_h ", max_h)
        print("22222222222222222222222222222222222222222222222")
        base = np.zeros((int(size[1] * x_number) + 8, int(size[1] * x_number) + 8, int(H_number)))  # 創出一個最大的base
        base[4:int(size[1] * x_number) + 4, 4:int(size[1] * x_number) + 4] = cube  # 左跟下拉出4格位置，然後把cube貼上去
        base[0:4, :, 0:20] = 1  # 左側建出20單位高的邊
        base[int(size[1] * x_number) + 4 : int(size[1] * x_number) + 8, :, 0:20] = 1  # 右側建出20單位高的邊
        base[:, 0:4, 0:20] = 1  # 上側建出20單位高的邊
        base[:, int(size[1] * x_number) + 4 : int(size[1] * x_number) + 8, 0:20] = 1  # 下側建出20單位高的邊

        base_1 = np.ones((int(size[1] * x_number) + 8, int(size[1] * x_number) + 8, 4))  # 再建出一個高4單位的綜合底
        #print(base_1.shape)
        #print(base.shape)
        base = np.concatenate((base_1, base), axis = 2)  # 把綜合底接在所有東西下面
       
        

        #base = find_surface_points(base)


        np.save("param_4000epoch_{}.npy".format(nn), base)  # 存成npy檔
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.96)
    parser.add_argument('--whether-load-model', type=bool, default=True, help="whether need to continus")
    parser.add_argument('--start-epoch', type=int, default=4000, help='which epoch')

    parser.add_argument('--root_path', type=str, default=r"C:\Users\Dennis\Desktop\SHW\sub_THz\Diffractive_deep_neural_network\data")
    parser.add_argument('--model-name', type=str, default='_model.pth')
    parser.add_argument('--model-save-path', type=str, default="./saved_model1/")
    parser.add_argument('--result-record-path', type=pathlib.Path, default="./result.csv", help="saving path")

    torch.backends.cudnn.benchmark = True
    args_ = parser.parse_args()
    random.seed(args_.seed)
    np.random.seed(args_.seed)
    torch.manual_seed(args_.seed)
    main(args_)

