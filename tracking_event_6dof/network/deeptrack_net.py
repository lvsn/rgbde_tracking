import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase
import numpy as np


class Fire(nn.Module):
    """
    From SqueezeNet : https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py

    """

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class DeepTrackNetEvent(NetworkBase):
    def __init__(self, image_size=150, split=2, channel_in=None):
        super(DeepTrackNetEvent, self).__init__()

        if not channel_in:
            channel_in = split*2

        filter_size_1 = 64
        embedding_size = 500
        half_filter_size_1 = int(filter_size_1 / 2)

        self.convA = nn.Conv2d(channel_in, filter_size_1, 3, 2)
        self.batchA = nn.BatchNorm2d(filter_size_1)
        self.fireA = Fire(filter_size_1, half_filter_size_1,
                          half_filter_size_1, half_filter_size_1)
        self.batchA2 = nn.BatchNorm2d(filter_size_1 * 2)

        self.fire1 = Fire(filter_size_1 * 2, filter_size_1,
                          filter_size_1, filter_size_1)
        self.batch1 = nn.BatchNorm2d(filter_size_1 * 4)
        self.fire2 = Fire(filter_size_1 * 4, filter_size_1,
                          filter_size_1 * 2, filter_size_1 * 2)
        self.batch2 = nn.BatchNorm2d(filter_size_1 * 8)
        self.fire3 = Fire(filter_size_1 * 8, filter_size_1 * 2,
                          filter_size_1 * 2, filter_size_1 * 2)
        self.batch3 = nn.BatchNorm2d(filter_size_1 * 12)

        view_width = int(int(int(int(int((image_size - 2)/2)/2)/2)/2)/2)

        self.view_size = filter_size_1 * 12 * view_width * view_width
        self.fc1 = nn.Linear(self.view_size, embedding_size)
        self.fc_bn1 = nn.BatchNorm1d(embedding_size)
        self.fc2 = nn.Linear(embedding_size, 6)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout_A0 = nn.Dropout2d(0.3)
        self.dropout_A1 = nn.Dropout2d(0.3)

    def forward(self, A):
        A = self.convA(A)
        A = self.batchA(F.elu(A, inplace=True))
        A = self.batchA2(F.max_pool2d(torch.cat((self.fireA(A), A), 1), 2))

        A = self.dropout_A0(self.batch1(
            F.max_pool2d(torch.cat((self.fire1(A), A), 1), 2)))
        A = self.dropout_A1(self.batch2(
            F.max_pool2d(torch.cat((self.fire2(A), A), 1), 2)))
        A = self.batch3(F.max_pool2d(torch.cat((self.fire3(A), A), 1), 2))

        A = A.view(-1, self.view_size)
        A = self.dropout1(A)
        A = self.fc_bn1(F.elu(self.fc1(A)))
        A = torch.tanh(self.fc2(A))
        return A

    def loss(self, predictions, targets):
        return nn.MSELoss()(predictions[0], targets[0])


class TemporalConvolution(NetworkBase):
    def __init__(self, kernel_size=5, image_size=150, learnable=True):
        super(TemporalConvolution, self).__init__()

        kernel_size = kernel_size
        kernel = self._gaussian_kernel(kernel_size, 0.5)

        self.image_size = image_size
        channels_size = image_size**2

        self.conv = nn.Conv3d(in_channels=1,
                              out_channels=1, padding=(kernel_size//2, 0, 0),
                              kernel_size=(kernel_size, 1, 1), groups=1, bias=False)

        self.conv.weight.data = kernel.float()
        self.conv.weight.requires_grad = learnable

    def _gaussian_kernel(self, kernel_size, sigma):
        x_coord = np.arange(kernel_size)

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        gaussian_kernel = (1./(2.*np.pi*variance)) *\
            np.exp(
            -(x_coord - mean)**2 / (2*variance)
        )
        gaussian_kernel = torch.tensor(gaussian_kernel)
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        return gaussian_kernel.view(1, 1, kernel_size, 1, 1)

    def __call__(self, data):
        data = data.unsqueeze(1)
        data = self.conv(data)
        data = data.squeeze(1)
        return data


class DeepTrackNetSpike(DeepTrackNetEvent):
    def __init__(self, image_size=150, channel_in=None, split=None):
        super(DeepTrackNetSpike, self).__init__(
            image_size=image_size, channel_in=channel_in)
        self.conv_temporal = TemporalConvolution()

    def forward(self, A):
        A = self.conv_temporal(A)
        return super().forward(A)
