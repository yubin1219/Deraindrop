import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision.models.vgg import vgg19
import numpy as np

class GANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.loss = nn.BCELoss().to(device)
        self.lsgan = nn.MSELoss().to(device)

    def convert_tensor(self, input, is_real):
        if is_real:
            return Variable(torch.FloatTensor(input.size()).fill_(self.real_label)).to(device)
        else:
            return Variable(torch.FloatTensor(input.size()).fill_(self.fake_label)).to(device)

    def __call__(self, input, is_real):
        return self.lsgan(input, self.convert_tensor(input,is_real).to(device))

class AttentionLoss(nn.Module):
    def __init__(self, theta=0.9, iteration=4):
        super(AttentionLoss, self).__init__()
        self.theta = theta
        self.iteration = iteration
        self.loss = nn.MSELoss().to(device)

    def __call__(self, A_, M_):
        loss_ATT = None
        for i in range(1, self.iteration+1):
            if i == 1:
                loss_ATT = pow(self.theta, float(self.iteration-i)) * self.loss(A_[i-1],M_)
            else:
                loss_ATT += pow(self.theta, float(self.iteration-i)) * self.loss(A_[i-1],M_)
        return loss_ATT

# VGG19 pretrained on Imagenet
def trainable_(net, trainable):
    for param in net.parameters():
        param.requires_grad = trainable

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.model = (vgg19(pretrained = True).to(device))
        trainable_(self.model, False)
        self.loss = nn.MSELoss().to(device)                 
        self.vgg_layers = self.model.features
        self.layer_names = { '0' : 'conv1_1', '3' : 'relu1_2', '6' : 'relu2_1', '8' : 'relu2_2',  '17' : 'relu3_4' }

    def get_layer_output(self, x):
      output = []
      for name, module in self.vgg_layers._modules.items():
        if isinstance(module, nn.ReLU):
          module = nn.ReLU(inplace=False)
        x = module(x)
        if name in self.layer_names:
          output.append(x)
      return output

    def get_GTlayer_output(self, x):
      with torch.no_grad():
        output = []
        for name, module in self.vgg_layers._modules.items():
          if isinstance(module, nn.ReLU):
            module = nn.ReLU(inplace=False)
          x = module(x)
          if name in self.layer_names:
            output.append(x)
      return output

    def __call__(self, O_, T_):
        o = self.get_layer_output(O_)
        t = self.get_GTlayer_output(T_)
        loss_PL = 0
        for i in range(len(t)):
            if i ==0:
                loss_PL = self.loss(o[i],t[i])
            else:
                loss_PL += self.loss(o[i],t[i])

        loss_PL=loss_PL/float(len(t))        
        loss_PL = Variable(loss_PL,requires_grad=True)
        return loss_PL
        
class MultiscaleLoss(nn.Module):
    def __init__(self, ld=[0.7 , 0.8 , 1.0], batch=1):
        super(MultiscaleLoss, self).__init__()
        self.loss = nn.L1Loss().to(device)
        self.ld = ld
        self.batch=batch

    def __call__(self, S_, gt):
        T_ = []
        for i in range(S_[0].shape[0]):
            temp = []
            x = (np.array(gt[i])*255.).astype(np.uint8)
            # print (x.shape, x.dtype)
            t = cv2.resize(x, None, fx=1.0/4.0,fy=1.0/4.0, interpolation=cv2.INTER_AREA)
            t = np.expand_dims((t/255.).astype(np.float32).transpose(2,0,1),axis=0)
            temp.append(t)
            t = cv2.resize(x, None, fx=1.0/2.0,fy=1.0/2.0, interpolation=cv2.INTER_AREA)
            t = np.expand_dims((t/255.).astype(np.float32).transpose(2,0,1),axis=0)
            temp.append(t)
            x = np.expand_dims((x/255.).astype(np.float32).transpose(2,0,1),axis=0)
            temp.append(x)
            T_.append(temp)
        temp_T = []
        for i in range(len(self.ld)):
            for j in range((S_[0].shape[0])):
                if j == 0:
                    x = T_[j][i]
                else:
                    x = np.concatenate((x, T_[j][i]), axis=0)
            temp_T.append(Variable(torch.from_numpy(x)).to(device))
        T_ = temp_T
        loss_ML = None
        for i in range(len(self.ld)):
            if i == 0: 
                loss_ML = self.ld[i] * self.loss(S_[i], T_[i])
            else:
                loss_ML += self.ld[i] * self.loss(S_[i], T_[i])
        
        return loss_ML/float(S_[0].shape[0])

class MaskLoss(nn.Module):
  def __init__(self):
    super(MaskLoss, self).__init__()
    self.loss = nn.L1Loss().to(device)

  def __call__(self, O, gt, M):
    O_M = O * M
    gt_M = gt * M
    return self.loss(O_M, gt_M)

class MAPLoss(nn.Module):
  def __init__(self, gamma=0.2):
    super(MAPLoss, self).__init__()
    self.loss = nn.MSELoss().to(device)
    self.gamma = gamma

  # D_map_O, D_map_R
  def __call__(self, D_O, D_R, I_, gt):
    x = (np.array(gt[0])*255.).astype(np.uint8)
    t = cv2.resize(x, None, fx=1.0/4.0,fy=1.0/4.0, interpolation=cv2.INTER_AREA)
    gt_ = (t/255.).astype(np.float32)
    x = (np.array(I_[0])*255.).astype(np.uint8)
    t = cv2.resize(x, None, fx=1.0/4.0,fy=1.0/4.0, interpolation=cv2.INTER_AREA)
    IN = (t/255.).astype(np.float32)
    M = get_mask(IN, gt_) #28,28,1
    M = [M]
    M = torch.from_numpy(np.array(M).transpose((0,3,1,2))).to(device)
    Z = Variable(torch.zeros(D_R.shape)).to(device)
    D_A = self.loss(D_O, M)
    D_Z = self.loss(D_R, Z)
    return self.gamma * (D_A + D_Z)
