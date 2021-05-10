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
    for i in range(4):
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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()        
         ##  Autoencoder  ##
        self.convm = nn.Sequential(
            nn.Conv2d(4, 32, 1, 1, 0)
            )
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=0)
            )
        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True)
            )
        self.encoder2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
            )
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0)
            )
        self.conv1_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, stride=1, padding=0),
            nn.ReLU(True)
            )
        self.conv1_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, stride=1, padding=0),
            nn.Sigmoid()
            )
        self.lrelu1 = nn.Sequential(
            nn.LeakyReLU(0.2,True)
            )
        self.dilconv1 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(256, 256, 3, padding=0, dilation=2),
            )
        self.conv2_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, stride=1, padding=0),
            nn.ReLU(True)
            )
        self.conv2_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, stride=1, padding=0),
            nn.Sigmoid()
            )
        self.lrelu2 = nn.Sequential(
            nn.LeakyReLU(0.2,True)
            )
        self.dilconv2 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(256, 256, 3, padding=0, dilation=4),
            )
        self.conv3_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, stride=1, padding=0),
            nn.ReLU(True)
            )
        self.conv3_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, stride=1, padding=0),
            nn.Sigmoid()
            )
        self.lrelu3 = nn.Sequential(
            nn.LeakyReLU(0.2,True)
            )
        self.dilconv3 = nn.Sequential(
            nn.ReflectionPad2d(8),
            nn.Conv2d(256, 256, 3, padding=0, dilation=8),
            )
        self.conv4_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, stride=1, padding=0),
            nn.ReLU(True)
            )
        self.conv4_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, stride=1, padding=0),
            nn.Sigmoid()
            )
        self.lrelu4 = nn.Sequential(
            nn.LeakyReLU(0.2,True)
            )
        self.dilconv4 = nn.Sequential(
            nn.ReflectionPad2d(16),
            nn.Conv2d(256, 256, 3, padding=0, dilation=16),
            )
        self.conv5_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, stride=1, padding=0),
            nn.ReLU(True)
            )
        self.conv5_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, stride=1, padding=0),
            nn.Sigmoid()
            )
        self.lrelu5 = nn.Sequential(
            nn.LeakyReLU(0.2,True)
            )
        self.outframe1 = nn.Sequential(
            nn.Conv2d(512, 3, 1, 1, 0)
            )
        ## upsample
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2,True)
            )
        self.conv6 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(384, 128, 3, 1, 0),
            nn.LeakyReLU(0.2,True)
            )
        self.outframe2 = nn.Sequential(
            nn.Conv2d(128, 3, 1, 1, 0)
            )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2,True)
            )
        self.conv7 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(96, 32, 3, 1, 0),
            nn.LeakyReLU(0.2,True)
            )
        self.output = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 3, 3, 1, 0)
            )
        
    def forward(self, input):      
        x = self.conv(input)
        input0 = x
        x = self.encoder1(x)
        input1 = x
        x = self.encoder2(x)
        
        input1 = x
        x = self.conv1(x)
        res = x
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = res * x
        x = self.lrelu1(x)
        x = input1 + x
        
        input2 = x
        x = self.dilconv1(x)
        res = x
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = res * x
        x = self.lrelu2(x)
        x = input2 + x
        input3 = x
        x = self.dilconv2(x)
        res = x
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = res * x
        x = self.lrelu3(x)
        x = input3 + x
        
        input4 = x
        x = self.dilconv3(x)
        res = x
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = res * x
        x = self.lrelu4(x)
        x = input4 + x
        
        input5 = x
        x = self.dilconv4(x)
        res = x
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = res * x
        x = self.lrelu5(x)
        x = input5 + x
        
        frame1 = self.outframe1(x)
        
        x = self.decoder1(x)
        x = torch.cat((x, input1), 1)
        x = self.conv6(x)
        frame2 = self.outframe2(x)
        
        x = self.decoder2(x)
        x = torch.cat((x, input0), 1)
        x = self.conv7(x)
        x = self.output(x)
        
        return frame1, frame2, x
