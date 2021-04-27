import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.LeakyReLU(0.2,True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(16, 16, 3, 2, 1),
            nn.LeakyReLU(0.2,True)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 64, 3, 1, 1),
            nn.LeakyReLU(0.2,True)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2,True)
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2,True)
            )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2,True)
            )
        self.conv_mask = nn.Sequential(
            nn.Conv2d(128, 1, 3, 1, 1)
            )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 2, 1),
            nn.LeakyReLU(0.2,True)
            )
        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 2, 1),
            nn.LeakyReLU(0.2,True)
            )
        self.fc = nn.Sequential(
            nn.Linear(32 * 14 * 14, 1024),
            nn.Linear(1024, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        mask = self.conv_mask(x)
        x = self.conv7(x * mask)
        x = self.conv8(x)
        x=x.reshape(x.size(0), -1)
        return [mask, self.fc(x)]
