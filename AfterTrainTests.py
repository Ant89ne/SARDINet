from DataLoad import *
from utils import *
from TransNet import *

import torch
from torch import nn 
from torch.utils.data import DataLoader

import torchgeometry as geo

import argparse
import random

random.seed(25)

parser = argparse.ArgumentParser()
parser.add_argument('modelFile', type=str, help='Path to the model file to test')
args = parser.parse_args()

percentages = [0.9,0.1]
path = "/media/bralet/Elements/DataBis/SpaceNet6/3DShapes/"
pathInput = "/media/bralet/Elements/DataBis/SpaceNet6/Norm/"
pathGT = "/media/bralet/Elements/DataBis/SpaceNet6/3DShapes/AOI_Subset_RGBNIR/"

model = torch.load(args.modelFile, map_location="cpu")

d_train = ImDataset(path, 400,800,1, percentages, 1, pathInput = pathInput, pathOutput = pathGT)
dataloader = DataLoader(d_train, batch_size=1)

loss_fn = nn.MSELoss()#reduction = "none")

# d,l = next(iter(dataloader))


# with torch.no_grad():
#     for i in range(len(d_train.SarImFiles)):
#         d = torch.Tensor(np.array(Image.open(d_train.SarImFiles[i]))).reshape(1,450,900,4)
#         d = d.transpose(1,3)
#         d /= 256.0
#         d = d[:,:, :800,:400]

#         l = torch.Tensor(np.array(Image.open(d_train.OptImFiles[i]))).reshape(1,450,900,4)
#         l /= 256.0
#         l = l[:,:400,:800,:]
#         pred = model(d)
#         pred = pred.transpose(1,3)
#         print(torch.max(pred), torch.min(pred))

#         p = torch.clone(pred)
#         pred[0,:,:,0] = p[0,:,:,2]
#         pred[0,:,:,2] = p[0,:,:,0]
#         lb = torch.clone(l)
#         l[0,:,:,0] = lb[0,:,:,2]
#         l[0,:,:,2] = lb[0,:,:,0]

#         plt.figure()
#         plt.subplot(211)
#         plt.imshow(pred[0,:,:,:3])
#         plt.subplot(212)
#         plt.imshow(l[0,:,:,:3])
#         plt.show()

#         print(getMoreMetrics(pred, l))


with torch.no_grad():
    # eval_loop(dataloader, model, loss_fn, "cpu")
    visIm(model, d_train, 400, "/home/bralet/Code/Translation/ResNormClas400/", 30, saveSep = True)

# plt.figure()
# plt.subplot(221)
# plt.hist(l[0,0,:,:], bins=256)
# plt.subplot(222)
# plt.hist(l[0,1,:,:], bins=256)
# plt.subplot(223)
# plt.hist(l[0,2,:,:], bins = 256)
# plt.subplot(224)
# plt.hist(l[0,3,:,:], bins = 256)
# plt.show()
# exit()


# p = model(d)
# plt.figure()
# p = torch.transpose(p, 1,3)
# l = torch.transpose(l, 1,3)

# with torch.no_grad():
#     plt.subplot(221)
#     plt.imshow(p[0,:,:,0], cmap="gray")
#     plt.colorbar()
#     plt.subplot(222)
#     plt.imshow(p[0,:,:,1], cmap="gray")
#     plt.colorbar()
#     plt.subplot(223)
#     plt.imshow(p[0,:,:,2], cmap="gray")
#     plt.colorbar()
#     plt.subplot(224)
#     plt.imshow(p[0,:,:,3], cmap="gray")
#     plt.colorbar()
#     print(torch.sum(p<0))
#     print(torch.min(torch.min(p, dim=1)[0], dim = 1))
#     print(torch.max(torch.max(p, dim=1)[0], dim = 1))
#     print(torch.sum(torch.sum(p, dim = 1)[0], dim = 0))
    
#     plt.figure()
#     plt.subplot(121)
#     plt.imshow(np.array(p[0,:,:,0:3]))#.astype(int))
#     plt.subplot(122)
#     plt.imshow(np.array(l[0,:,:,0:3]))#.astype(int))
    
#     plt.show()


# exit()
# ssim = loss_fn(p,l)
# with torch.no_grad():
#     plt.figure()
#     plt.subplot(121)
#     plt.imshow(ssim[0,0,:,:])
#     plt.colorbar()
#     plt.subplot(222)
#     plt.imshow(p[0,0,:,:])
#     plt.subplot(224)
#     plt.imshow(l[0,0,:,:])
#     print(torch.mean(ssim[0,0,:,:]))
#     plt.show()

# print(loss_fn)




# reconstructFull(d_train, model, 10)




# loss_fn = nn.L1Loss()

# with torch.no_grad():

#     lo = 0
#     for d,l in dataloader :
#         p = model(d)
#         loss = loss_fn(p,l).item() #torch.mean(loss_fn(p,l) * l).item()
#         print(loss, torch.max(l), torch.mean(l), torch.median(l))
#         lo += loss
#         plt.figure()
#         plt.imshow(l[0,0,:,:]-p[0,0,:,:])
#         plt.show()
        
#     print(lo, lo/len(dataloader))

#     print(torch.max(d), torch.max(l), torch.max(p))
#     print(torch.min(d), torch.min(l), torch.min(p))

#     print(loss_fn(p,l))
    # p = torch.transpose(p,1,3)
    # p +=0.5
        
    # eval_loop(dataloader, model, loss_fn, "cpu")