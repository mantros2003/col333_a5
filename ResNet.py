from torch import nn

class ResNet(nn.Module):
    def __init__(self, num_channels, out_channels, kernel_sz) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.out_channels = out_channels
        self.kernel_sz = kernel_sz

        self.net = nn.Sequential(*[
            nn.Conv2d(self.num_channels, self.out_channels, self.kernel_sz),
            nn.ReLU(),
            nn.Conv2d(self.num_channels, self.out_channels, self.kernel_sz)
        ])
    
    def forward(self, x):
        return self.net(x) + x