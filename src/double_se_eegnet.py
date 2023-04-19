import torch
from torch import nn
from torch.nn.functional import elu

from braindecode.models.modules import Expression, Ensure4d
from braindecode.models.functions import squeeze_final_output
from braindecode.models.eegnet import Conv2dWithConstraint, _transpose_to_b_1_c_0, _transpose_1_0


class DoubleSEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(DoubleSEBlock, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation_wo_bias = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        res = x
        x = self.squeeze(x)
        cse = self.excitation_wo_bias(x)
        sse = self.excitation(x)
        out = cse.expand_as(res) * res + sse.expand_as(res) * res
        return out


# modified from braindecode
class DoubleSEEEGNet(nn.Sequential):
    def __init__(
        self,
        in_chans,
        n_classes,
        input_window_samples=None,
        final_conv_length="auto",
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        third_kernel_size=(8, 4),
        drop_prob=0.25,
    ):
        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.third_kernel_size = third_kernel_size
        self.drop_prob = drop_prob

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.add_module("ensuredims", Ensure4d())
        self.add_module("dimshuffle", Expression(_transpose_to_b_1_c_0))

        self.add_module(
            "conv_temporal",
            nn.Conv2d(
                1,
                self.F1,
                (1, self.kernel_length),
                stride=1,
                bias=False,
                padding=(0, self.kernel_length // 2),
            ),
        )
        self.add_module(
            "bnorm_temporal",
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
        )
        self.add_module('SELayer_1', DoubleSEBlock(self.F1, reduction_ratio=8))
        self.add_module(
            "conv_spatial",
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (self.in_chans, 1),
                max_norm=1,
                stride=1,
                bias=False,
                groups=self.F1,
                padding=(0, 0),
            ),
        )

        self.add_module(
            "bnorm_1",
            nn.BatchNorm2d(
                self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3
            ),
        )
        self.add_module('SELayer_2', DoubleSEBlock(self.F1 * self.D, reduction_ratio=16))
        self.add_module("elu_1", Expression(elu))

        self.add_module("pool_1", pool_class(kernel_size=(1, 4), stride=(1, 4)))
        self.add_module("drop_1", nn.Dropout(p=self.drop_prob))

        self.add_module(
            "conv_separable_depth",
            nn.Conv2d(
                self.F1 * self.D,
                self.F1 * self.D,
                (1, 16),
                stride=1,
                bias=False,
                groups=self.F1 * self.D,
                padding=(0, 16 // 2),
            ),
        )
        self.add_module(
            "conv_separable_point",
            nn.Conv2d(
                self.F1 * self.D,
                self.F2,
                (1, 1),
                stride=1,
                bias=False,
                padding=(0, 0),
            ),
        )

        self.add_module(
            "bnorm_2",
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
        )
        
        self.add_module("elu_2", Expression(elu))
        self.add_module("pool_2", pool_class(kernel_size=(1, 8), stride=(1, 8)))
        self.add_module("drop_2", nn.Dropout(p=self.drop_prob))

        out = self(
            torch.ones(
                (1, self.in_chans, self.input_window_samples, 1),
                dtype=torch.float32
            )
        )
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        if self.final_conv_length == "auto":
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        self.add_module('SELayer_3', DoubleSEBlock(self.F2, reduction_ratio=16))
        self.add_module(
            "conv_classifier",
            nn.Conv2d(
                self.F2,
                self.n_classes,
                (n_out_virtual_chans, self.final_conv_length),
                bias=True,
            ),
        )
        self.add_module("softmax", nn.LogSoftmax(dim=1))
        # Transpose back to the the logic of braindecode,
        # so time in third dimension (axis=2)
        self.add_module("permute_back", Expression(_transpose_1_0))
        self.add_module("squeeze", Expression(squeeze_final_output))
