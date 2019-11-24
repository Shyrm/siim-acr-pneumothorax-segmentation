import torch
from torch import nn
from torch.nn import functional as F
import pretrainedmodels
from torchsummary import summary

from fastai.layers import SequentialEx, SigmoidRange, conv2d, PixelShuffle_ICNR, conv_layer, batchnorm_2d
from fastai.callbacks.hooks import model_sizes, hook_outputs, Hook, _hook_inner, dummy_eval
from fastai.vision.models.unet import _get_sfs_idxs

from Architectures.fast_ai_unet import create_body_local


nonlinearity = nn.ReLU


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, n_filters, hook, is_deconv=False, scale=True):
        super().__init__()

        self.hook = hook
        self.shuf = nn.PixelShuffle(upscale_factor=2)
        self.norm = nn.BatchNorm2d(n_filters)
        self.relu = nonlinearity(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)

        if scale:
            # B, C/4, H, W -> B, C/4, H, W
            if is_deconv:
                self.upscale = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                                  stride=2, padding=1, output_padding=1)
            else:
                self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upscale = nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=1)

        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):

        s = self.hook.stored
        x = self.shuf(x)
        ssh = s.shape[-2:]
        if ssh != x.shape[-2:]:
            x = F.interpolate(x, s.shape[-2:], mode='nearest')
        x = self.relu(torch.cat([x, self.norm(s)], dim=1))

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.upscale(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return x


class Hcolumns(nn.Module):
    def __init__(self, hooks, nc):
        super(Hcolumns, self).__init__()
        self.hooks = hooks
        self.n = len(self.hooks)
        self.factorization = None
        if nc is not None:
            self.factorization = nn.ModuleList()
            for i in range(self.n):
                self.factorization.append(nn.Sequential(
                    conv2d(nc[i], nc[-1], 3, padding=1, bias=True),
                    conv2d(nc[-1], nc[-1], 3, padding=1, bias=True)))

    def forward(self, x):
        n = len(self.hooks)

        out = [F.interpolate(self.hooks[i].stored if self.factorization is None
                             else self.factorization[i](self.hooks[i].stored), scale_factor=2 ** (self.n - i),
                             mode='bilinear', align_corners=False) for i in range(self.n)] + [x]

        return torch.cat(out, dim=1)


class FinalBlock(nn.Module):
    def __init__(self, in_channels, last_filters, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, last_filters, 3, padding=1)
        self.relu1 = nonlinearity(inplace=True)
        self.conv2 = nn.Conv2d(last_filters, num_classes, 3, padding=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        return x


class ResNet34Hyper(SequentialEx):

    def __init__(self, encoder=None, n_classes=2, last_filters=32, imsize=(256, 256), y_range=None, **kwargs):

        self.n_classes = n_classes

        layers = nn.ModuleList()

        # Encoder
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
        layers.append(encoder)

        x = dummy_eval(encoder, imsize).detach()

        self.hc_hooks = []
        hc_c = []

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(conv_layer(ni, ni * 2),
                                    conv_layer(ni * 2, ni)).eval()
        x = middle_conv(x)
        layers.extend([batchnorm_2d(ni), nn.ReLU(), middle_conv])

        # self.hc_hooks = [Hook(layers[-1], _hook_inner, detach=False)]
        # hc_c = [x.shape[1]]

        # Decoder
        n_filters = [64, 128, 256, 512]
        n = len(n_filters)
        is_deconv = True

        for i, idx in enumerate(sfs_idxs[:-1]):
            in_c, out_c = int(n_filters[n - i - 1] + n_filters[n - i - 2]) // 2, int(sfs_szs[idx][1])

            dec_bloc = DecoderBlock(in_c, out_c, self.sfs[i], is_deconv, True).eval()
            layers.append(dec_bloc)

            x = dec_bloc(x)

            self.hc_hooks.append(Hook(layers[-1], _hook_inner, detach=False))
            hc_c.append(x.shape[1])

        ni = x.shape[1]

        layers.append(PixelShuffle_ICNR(n_filters[0], scale=2))

        layers.append(Hcolumns(self.hc_hooks, hc_c))

        fin_block = FinalBlock(ni * (len(hc_c) + 1), last_filters, n_classes)
        layers.append(fin_block)

        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)


if __name__ == '__main__':

    arch = pretrainedmodels.resnet34

    body = create_body_local(
        arch=arch,
        pretrained=True,
        cut=None)  # extract encoder part from encoder network

    body = body if len(body) > 1 else body[0]

    model = ResNet34Hyper(encoder=body, n_classes=2)
    summary(model, (3, 256, 256), device='cpu')
