# ===============
# 這是一段 從訓練好的光學神經網路 (D2NN / ONN) 模型中取出各層參數，並轉換成實際可製作的三維光學結構資料 (3D voxel model) 的程式。
# 它的目的不是訓練模型，而是 根據模型的 phase pattern 產生對應的 3D 幾何結構 (.npy 檔案)，讓你之後能做模擬或實體實作。
# ===============
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)

import argparse
import numpy as np
import torch
from torch.nn import functional as F
import math
from model.opticalSimulation import ONN, MaterialLayer # import your model here
from model.autoencoder import Autoencoder
from model.restormer250724 import Restormer
from config import AUTOENCODER_CONFIG, ENCODER_CONFIG, RESTORMER_CONFIG



def find_surface_points(data):
    # Ensure the data is treated as a boolean array
    data = data.astype(bool)
    
    # Create an empty boolean array of the same shape as the input
    surface = np.zeros_like(data, dtype=bool)

    # Check each point in the array, avoiding the edges to prevent index errors
    for i in range(1, data.shape[0] - 1):
        for j in range(1, data.shape[1] - 1):
            for k in range(1, data.shape[2] - 1):
                # A point is considered a surface point if it is True and at least one of its neighbors is False, then it will be a surface point
                if data[i, j, k] and not data[i-1:i+2, j-1:j+2, k-1:k+2].all():
                    surface[i, j, k] = True

    # Handle edges to ensure surface definition includes the boundary points
    # First layer and last layer along each axis
    surface[0, :, :], surface[-1, :, :] = data[0, :, :], data[-1, :, :]
    surface[:, 0, :], surface[:, -1, :] = data[:, 0, :], data[:, -1, :]
    surface[:, :, 0], surface[:, :, -1] = data[:, :, 0], data[:, :, -1]

    return surface

def weight2npy(args):
    x_size = 0.00040   # lateral fabrication resolution (m)
    dx = 0.00075  # pixel size (m) which equals to ONN pixel length
    x_number = int(dx/x_size)  

    vol_size = 0.000025  # vertical fabrication resolution (m)
    H_thickness = 0.003  # maximum vertical (m)
    H_number = int(H_thickness/vol_size) # 120

    n = 1.70  # material refractive index
    #n = 3.42
    dn = n - 1  # effective refractive index contrast
    c = 2.998e8 
    lam = c / 0.2004e12  # wavelength which should be identical with experiments
    k_air = 2 * math.pi / lam  # wave number
    k_mat = 2 * math.pi / lam * n

    # make output dirs
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    encoder = ONN(ENCODER_CONFIG)
    decoder = Restormer(RESTORMER_CONFIG)
    model = Autoencoder(encoder, decoder, AUTOENCODER_CONFIG)  # load your model here
    print(model)
    print(model.encoder)
    model.cuda()  # move to cuda
    path = os.path.join(args.weights_dir, args.weights_name)
    # weights = torch.load(path)
    # print(weights.keys())
    model.encoder.load_state_dict(torch.load(path), strict=False)  # load your weight here

    with torch.no_grad():
        for layer in model.encoder.layers:
            if isinstance(layer, MaterialLayer):
                for name, param in layer.named_parameters():
                    
                    size = param.size()
                    #param = 2 * math.pi * torch.nn.Sigmoid(param)  # 用 Sigmoid 將參數壓到 (0, 1)，較粗糙的轉換方式
                    param = torch.remainder(param, 2 * math.pi)  # 用 module 將參數取到 (0, 1)，較細緻的轉換方式，需要搭配正確的training過程
                    print("Layer name:", name, ", Layer shape:", param.shape)

                    # 用 manufacture voxel 為單位定義一片 ONN 的長寬高
                    cube = np.zeros((int(size[0] * x_number), int(size[1] * x_number), H_number))  #! 用 int() 轉換感覺會有很多多於長度沒有考慮的問題？
                    print("min: ", torch.min(param), "max: ", torch.min(param))

                    min_h = 0.00000  # 最小限制厚度
                    max_h = 1.00000  # 最大限制厚度
                    for x in range(size[0]):
                        for y in range(size[1]):
                            # h = param[x, y] / k_air / dn  # 相位轉成厚度, radian / (radian/length) / (折射率) = length #! 似乎算錯了
                            h = param[x, y] / k_mat / dn  # 0 ~ 1.2605e-3 #! 感覺這個才對 

                            if (float(h) > max_h):  # 更新最大限制
                                max_h = float(h)
                            if (float(h) < min_h):  # 更新最小限制
                                min_h = float(h)

                            h = int(torch.round(h / vol_size))  # quantize 到 manufacture voxel # 50.42
                            
                            plane = np.ones((x_number, x_number, h))  # 一個單位柱子的大小長相
                            cube[0 + x_number * x : 0 + x_number * (x + 1), 0 + x_number * y : 0 + x_number * (y + 1), : h] = plane  # 一次填一個 ONN pixel，把需要製造的單位柱子填成1，否則是0

                    #np.save("param_pixel_{}.npy".format(nn), cube)
                    
                    # print("min_h ", min_h, "max_h ", max_h)
                    
                    base = np.zeros((int(size[1] * x_number) + 8, int(size[1] * x_number) + 8, H_number))  # 創出一個長跟寬都在更大8個manufacture pixel的
                    base[4:int(size[1] * x_number) + 4, 4:int(size[1] * x_number) + 4, :] = cube  # 長寬各空出4格位置，然後把cube貼上去
                    base[0:4, :, 0:20] = 1  # 左側建出20單位高的邊
                    base[int(size[1] * x_number) + 4 : int(size[1] * x_number) + 8, :, 0:20] = 1  # 右側建出20單位高的邊
                    base[:, 0:4, 0:20] = 1  # 上側建出20單位高的邊
                    base[:, int(size[1] * x_number) + 4 : int(size[1] * x_number) + 8, 0:20] = 1  # 下側建出20單位高的邊

                    base_1 = np.ones((int(size[1] * x_number) + 8, int(size[1] * x_number) + 8, 4))  # 再建出一個高4單位的綜合底
                    #print(base_1.shape)
                    #print(base.shape)
                    base = np.concatenate((base_1, base), axis = 2)  # 把綜合底接在所有東西下面
                    #base = find_surface_points(base)
                    #np.save("test.npy".format(nn), base)  # 存成npy檔
                    save_path = os.path.join(args.output_dir, args.output_name)
                    np.save(save_path, base)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-dir', type=str, default='./checkpoints_weights/baseline_restormer_ONN/weights')
    parser.add_argument('--weights-name', type=str, default='epoch60_valLoss0.0021_20251030_224229.pth')
    parser.add_argument('--output-dir', type=str, default='./ONN_manufacture', help='directory for saving npy files')
    parser.add_argument('--output-name', type=str, default='test', help='output file name')
    args_ = parser.parse_args()

    weight2npy(args_)
