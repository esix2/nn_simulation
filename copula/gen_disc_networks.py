import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        kernel_size = 4
        self.main = nn.Sequential(
            nn.Conv2d(in_channel=noise_dim + 1, out_channels=64, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channel=64, out_channels=128, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channel=128,out_channels= 64, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channel=64, out_channels=1, kernel_size=kernel_size, stride=1, padding=1),
            nn.Tanh()
        )
        

    def forward(self, simulated_map, noise_map):
        # Concatenate simulated map and noise map along the channels dimension
#         print(f"noise_map size: {noise_map.size()}")
#         print(f"simulated_map size: {simulated_map.size()}")
        input = torch.cat((simulated_map, noise_map), 1)
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128,  kernel_size=4, stride=2, padding=1, bias=False),  # Output is (128, 64, 64)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size= 4, stride=2, padding=1, bias=False),  # Output is (256, 32, 32)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size= 4, stride=2, padding=1, bias=False),  # Output is (51stride=2, padding=16, 16)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False),  # Output is (1, 1, 1)
#             nn.Conv2d(512, 1, 2, 1, 0, bias=False),  # Output is (1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
