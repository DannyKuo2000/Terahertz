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


import D2NN_ONN_DNN_pretest_absorption_v0

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

    vol_size = 0.000025
    #vol_size = 0.000005
    x_size = 0.00040
    H_thickness = 0.003
    H_number = int(H_thickness/vol_size)
    dx = 0.00075
    x_number = int(dx/x_size)
    n = 1.70
    #n = 3.42
    dn = n -1
    c = 3e8
    lam = c/0.2e12
    k = 2*math.pi/lam

    nn = 0

    

    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    model = D2NN_ONN_DNN_pretest_absorption_v0.Net()
    model.cuda()

    model.load_state_dict(torch.load(args.model_save_path + str(args.start_epoch) + args.model_name))


    for name,param in model.named_parameters():

        #print(torch.exp(i*param))
        #print("+++++++++++++++++++++++")
        
        size = param.size()
        sig = torch.nn.Sigmoid()
        param = sig(param)
        param = param*2*math.pi
        
        print("+++++++++++++++++++++++++")
        print(param)
        print(param.shape)
        print(name)
        print("+++++++++++++++++++++++++")
        
        #param = torch.sin(param)*math.pi
        #param = param + math.pi
        

        nn = nn +1


        #param = torch.sin(param)+1
        #param/=2
        #param = param*lam/dn
        #print(param)

        

        #nn = nn +1

        #print(torch.exp(i*(param)))


        cube = np.zeros((int(size[0]*x_number),int(size[1]*x_number),int(H_number)))
        d_max=lam/dn
        print("min ",torch.min(param),"max ",torch.min(param))
        min1=0.00000
        max1=0.00000


        for x in range(size[0]):
            for y in range(size[1]):

                h = param[x,y]/k/dn
                
                #print("333333333333333333333333333333",param[x,y],"333333333333333333333333333333")
                #h = param[x,y]*d_max
                if(float(h)>max1):
                    max1=float(h)
                if(float(h)<min1):
                    min1=float(h)


                h = int(torch.round(h/vol_size))
                

                #print(h)

                plane = np.ones((x_number, x_number, h))
                cube[0+x_number*x:0+x_number*(x+1), 0+x_number*y:0+x_number*(y+1),:h] = plane

        #np.save("param_pixel_{}.npy".format(nn), cube)
        print("22222222222222222222222222222222222222222222222")
        print("min1 ",min1,"max1 ",max1)
        print("22222222222222222222222222222222222222222222222")
        base = np.zeros((int(size[1]*x_number)+8,int(size[1]*x_number)+8,int(H_number)))
        base[4:int(size[1]*x_number)+4,4:int(size[1]*x_number)+4] = cube
        base[0:4,:,0:20] = 1
        base[int(size[1]*x_number)+4:int(size[1]*x_number)+8,:,0:20] = 1
        base[:,0:4,0:20] = 1
        base[:,int(size[1]*x_number)+4:int(size[1]*x_number)+8,0:20] = 1

        base_1 = np.ones((int(size[1]*x_number)+8,int(size[1]*x_number)+8,4))
        #print(base_1.shape)
        #print(base.shape)
        base = np.concatenate((base_1,base), axis = 2)
       
        

        #base = find_surface_points(base)


        np.save("param_4000epoch_{}.npy".format(nn), base)
                




                
        


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

