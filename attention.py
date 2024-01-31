import torch
import torch.nn as nn

class SABlock(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels*4, 3, padding=1, bias=False),
                                        nn.Sigmoid())
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        self.conv_6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False)
        self.conv_12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False)

    def forward(self, x):
        attention_mask = self.attention(x)
        features_1 = self.conv_1(x)
        features_3 = self.conv_3(x)
        features_6 = self.conv_6(x)
        features_12 = self.conv_12(x)

        out = torch.cat([features_1, features_3, features_6, features_12], 1) 
        return torch.mul(out, attention_mask)