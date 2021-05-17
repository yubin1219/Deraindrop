import torch
import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 64, 5, 1, 2),
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
    self.lrelu = nn.Sequential(
        nn.LeakyReLU(0.2)
        )
    self.conv2 = nn.Sequential(
        nn.Conv2d(64, 128, 3, 1, 1),
        nn.LeakyReLU(0.2)
        )
    self.conv_mask = nn.Sequential(
        nn.Conv2d(128, 1, 3, 1, 1)
        )
    self.conv3 = nn.Sequential(
        nn.Conv2d(128, 64, 5, 4, 1),
        nn.LeakyReLU(0.2)
        )
    self.conv4 = nn.Sequential(
        nn.Conv2d(64, 32, 5, 4, 1),
        nn.LeakyReLU(0.2)
        )
    self.fc = nn.Sequential(
        nn.Linear(32 * 14 * 14, 1024),
        nn.Linear(1024, 1),
        nn.Sigmoid()
        )

  def forward(self, x):
    x = self.conv1(x)
    x = self.res1(x) + x
    x = self.res2(x) + x
    x = self.res3(x) + x
    x = self.res4(x) + x
    x = self.lrelu(x)
    x = self.conv2(x)

    mask = self.conv_mask(x)

    x = x * mask
    x = self.conv3(x)
    x = self.conv4(x)
    x = x.reshape(x.size(0), -1)

    return [mask, self.fc(x)]
