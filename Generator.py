import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attmap_G(nn.Module):
  def __init__(self):
    super(Attmap_G, self).__init__()
    self.res0 = nn.Sequential(
        nn.Conv2d(4, 64, 3, 1, 1),
        nn.LeakyReLU(0.2)
        )
    self.res1 = nn.Sequential(
        nn.Conv2d(64, 32, 1, 1, 0),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 64, 1, 1, 0)
        )
    self.res2 = nn.Sequential(
        nn.Conv2d(64, 32, 1, 1, 0),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 64, 1, 1, 0)
        )
    self.res3 = nn.Sequential(
        nn.Conv2d(64, 32, 1, 1, 0),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 64, 1, 1, 0)
        )
    self.res4 = nn.Sequential(
        nn.Conv2d(64, 32, 1, 1, 0),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 64, 1, 1, 0)
        )
    self.shortcut = nn.Sequential(
        nn.Conv2d(64, 32, 1, 1, 0)
        )
    self.res5 = nn.Sequential(
        nn.Conv2d(64, 32, 1, 1, 0),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 32, 3, 1, 1)
        )
    self.conv_i = nn.Sequential(
        nn.Conv2d(32 + 32, 32, 3, 1, 1),
        nn.Sigmoid()
        )
    self.conv_f = nn.Sequential(
        nn.Conv2d(32 + 32, 32, 3, 1, 1),
        nn.Sigmoid()
        )
    self.conv_g = nn.Sequential(
        nn.Conv2d(32 + 32, 32, 3, 1, 1),
        nn.Tanh()
        )
    self.conv_o = nn.Sequential(
        nn.Conv2d(32 + 32, 32, 3, 1, 1),
        nn.Sigmoid()
        )
    self.det_conv_mask = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(32, 1, 3, 1, 0),
        nn.Sigmoid()
        )
    
  def forward(self, input):
    batch_size, row, col = input.size(0), input.size(2), input.size(3)
    mask = Variable(torch.ones(batch_size, 1, row, col)).to(device) / 2.
    h = Variable(torch.ones(batch_size, 32, row, col)).to(device) / 2.
    c = Variable(torch.ones(batch_size, 32, row, col)).to(device) / 5.
    mask_list = []
    for i in range(5):
      x = torch.cat((input, mask), 1)
      x = self.res0(x)
      x = self.res1(x) + x
      x = self.res2(x) + x
      x = self.res3(x) + x
      x = self.res4(x) + x
      shortcut = self.shortcut(x)
      x = F.leaky_relu(self.res5(x) + shortcut,negative_slope=0.2)

      x = torch.cat((x, h), 1)
      i = self.conv_i(x)
      f = self.conv_f(x)
      g = self.conv_g(x)
      o = self.conv_o(x)
      c = f * c + i * g
      h = o * torch.tanh(c)
      mask = self.det_conv_mask(h)
      mask_list.append(mask)

    x = torch.cat((input, mask), 1)

    return mask_list, x

# Shrink and Gating mechanism
class SGBlock(nn.Module):
  def __init__(self, dim = 256, reduction = 32, dilation = 1):
    super(SGBlock, self).__init__()
    self.conv1 = nn.Sequential(
        nn.LeakyReLU(0.2,True),
        nn.ReflectionPad2d(dilation),
        nn.Conv2d(dim, dim, 3, stride = 1, padding = 0, dilation = dilation)
        )
    self.conv2 = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, reduction, 3, 1, 0),
        nn.ReLU(True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(reduction, dim, 3, 1, 0),
        nn.Sigmoid()
        )
  def forward(self, x):
    res = self.conv1(x)
    out = self.conv2(res)
    out = out * res
    out = out + x

    return out
  
class Autoencoder_G(nn.Module):
  def __init__(self):
    super(Autoencoder_G, self).__init__()        
    ##  Autoencoder  ##
    self.conv = nn.Sequential(
      nn.ReflectionPad2d(1),
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=0)
            )
    self.encoder1 = nn.Sequential(
      nn.ReflectionPad2d(1),
      nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
      nn.ReLU(True),
      nn.ReflectionPad2d(1),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
      nn.ReLU(True)
      )
    self.encoder2 = nn.Sequential(
      nn.ReflectionPad2d(1),
      nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
      nn.ReLU(True),
      nn.ReflectionPad2d(1),
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
      nn.ReLU(True)
      )
    self.SG_c1 = SGBlock(dilation = 1)
    
    self.SG_d1 = SGBlock(dilation = 2)
    
    self.SG_d2 = SGBlock(dilation = 4)

    self.SG_d3 = SGBlock(dilation = 8)

    self.SG_d4 = SGBlock(dilation = 16)
    
    self.SG_c2 = SGBlock(dilation=1)
    
    self.lrelu6 = nn.Sequential(
      nn.LeakyReLU(0.2,True)
      )
    self.conv7 = nn.Sequential(
      nn.ReflectionPad2d(1),
      nn.Conv2d(256, 256, 3, 1, 0)
      )
    self.outframe1 = nn.Sequential(
      nn.Conv2d(256, 3, 1, 1, 0)
      )
        
    ## upsample
    self.decoder1 = nn.Sequential(
      nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
      nn.LeakyReLU(0.2,True)
      )
    self.conv8 = nn.Sequential(
      nn.ReflectionPad2d(1),
      nn.Conv2d(128, 64, 3, 1, 0),
      nn.LeakyReLU(0.2,True)
      )
    self.outframe2 = nn.Sequential(
      nn.Conv2d(64, 3, 1, 1, 0)
      )
        
    self.decoder2 = nn.Sequential(
      nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
      nn.LeakyReLU(0.2,True)
            )
    self.conv9 = nn.Sequential(
      nn.ReflectionPad2d(1),
      nn.Conv2d(64, 32, 3, 1, 0),
      nn.LeakyReLU(0.2,True)
      )
    self.output = nn.Sequential(
      nn.ReflectionPad2d(1),
      nn.Conv2d(32, 3, 3, 1, 0),
      nn.ReLU(True),
      nn.Tanh()
      )
        
  def forward(self, input):      
    x = self.conv(input)
    input0 = x
    x = self.encoder1(x)
    input1 = x
    x = self.encoder2(x)
        
    x = self.SG_c1(x)
    x = self.SG_d1(x)
    x = self.SG_d2(x)
    x = self.SG_d3(x)
    x = self.SG_d4(x)
    x = self.SG_c2(x)
        
    x = self.lrelu6(x)
    x = self.conv7(x)
    frame1 = self.outframe1(x)
        
    x = self.decoder1(x)
    x = torch.cat((x, input1), 1)
    x = self.conv8(x)
    frame2 = self.outframe2(x)
        
    x = self.decoder2(x)
    x = torch.cat((x, input0), 1)
    x = self.conv9(x)
    x = self.output(x)
        
    return frame1, frame2, x
