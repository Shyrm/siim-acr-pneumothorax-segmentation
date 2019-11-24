from fastai.vision import *
from fastai.callbacks.hooks import model_sizes, hook_outputs, dummy_eval, Hook, _hook_inner
from fastai.vision.models.unet import _get_sfs_idxs, UnetBlock
from fastai.vision.learner import cnn_config

import torchvision
from torchsummary import summary
import pretrainedmodels


class Hcolumns(nn.Module):
    def __init__(self, hooks: Collection[Hook], nc: Collection[int] = None, is_se_resnext=False):
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
                # self.factorization.append(conv2d(nc[i],nc[-1],3,padding=1,bias=True))
        self.is_se_resnext = is_se_resnext

    def forward(self, x: Tensor):
        n = len(self.hooks)
        out = [F.interpolate(self.hooks[i].stored if self.factorization is None
                             else self.factorization[i](self.hooks[i].stored), scale_factor=2 ** (self.n - i) * (2 if self.is_se_resnext else 1),
                             mode='bilinear', align_corners=False) for i in range(self.n)] + [x]
        return torch.cat(out, dim=1)


class AttentionBlock(nn.Module):

    def __init__(self, x_in_c:int, hook:Hook, final_div:bool=True, leaky:float=None):

        super(AttentionBlock, self).__init__()

        self.hook = hook

        g_out_c = x_in_c if final_div else x_in_c//2
        x_c = x_in_c if final_div else x_in_c // 2

        self.g1_conv = nn.Conv2d(x_in_c, g_out_c, kernel_size=1)
        self.g1_bn = nn.BatchNorm2d(g_out_c)

        self.x1_conv = nn.Conv2d(x_c, x_c, kernel_size=1)
        self.x1_bn = nn.BatchNorm2d(x_c)

        self.psi_conv = nn.Conv2d(x_c, 1, kernel_size=1)
        self.psi_bn = nn.BatchNorm2d(1)

        self.relu = relu(leaky=leaky)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x:Tensor) -> Tensor:
        s = self.hook.stored
        g1 = self.g1_bn(self.g1_conv(s))
        x1 = self.x1_bn(self.x1_conv(x))

        g1_x1 = torch.add(g1, x1)
        psi = self.relu(g1_x1)
        psi = self.psi_bn(self.psi_conv(psi))
        psi = self.sigmoid(psi)

        return torch.mul(x, psi)


class UnetBlockAttention(nn.Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."
    def __init__(self, up_in_c:int, x_in_c:int, hook:Hook, final_div:bool=True, blur:bool=False, leaky:float=None,
                 **kwargs):
        super(UnetBlockAttention, self).__init__()
        self.hook = hook
        self.attention_block = AttentionBlock(x_in_c, hook, final_div=final_div, leaky=leaky)
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, leaky=leaky, **kwargs)

    def forward(self, up_in:Tensor) -> Tensor:
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.attention_block(up_out)
        return cat_x


class DynamicUnetAttention_Hcolumns(SequentialEx):
    "Create a U-Net from a given architecture."

    def __init__(self, encoder: nn.Module, n_classes: int, imsize: Tuple = (224, 224),
                 blur: bool = False, blur_final=True,
                 self_attention: bool = False,
                 y_range: Optional[Tuple[float, float]] = None,
                 last_cross: bool = True, bottle: bool = False,
                 is_se_resnext = False, **kwargs):

        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(conv_layer(ni, ni * 2, **kwargs),
                                    conv_layer(ni * 2, ni, **kwargs)).eval()
        x = middle_conv(x)
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        self.hc_hooks = [Hook(layers[-1], _hook_inner, detach=False)]
        hc_c = [x.shape[1]]

        # layers = [encoder]

        # self.hc_hooks = []
        # hc_c = []

        for i, idx in enumerate(sfs_idxs):
            not_final = i != len(sfs_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])

            do_blur = blur and (not_final or blur_final)

            unet_block = UnetBlockAttention(up_in_c, x_in_c, self.sfs[i], final_div=not_final,
                                   blur=blur, **kwargs).eval()

            layers.append(unet_block)

            x = unet_block(x)

            print(x.shape)

            self.hc_hooks.append(Hook(layers[-1], _hook_inner, detach=False))
            hc_c.append(x.shape[1])

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, **kwargs))

        if is_se_resnext:
            layers.append(PixelShuffle_ICNR(ni, **kwargs))

        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, **kwargs))

        hc_c.append(ni)

        layers.append(Hcolumns(self.hc_hooks, hc_c, is_se_resnext=is_se_resnext))
        layers += [conv_layer(ni * len(hc_c), n_classes, ks=1, use_activ=False, **kwargs)]

        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()


def has_pool_type(m):
    if is_pool_type(m): return True
    for l in m.children():
        if has_pool_type(l): return True
    return False


def create_body_local(arch: Callable, pretrained: bool = True, cut: Optional[Union[int, Callable]] = None):
    "Cut off the body of a typically pretrained `model` at `cut` (int) or cut the model as specified by `cut(model)` (function)."

    if pretrained:
        model = arch(num_classes=1000, pretrained='imagenet')

    cut = ifnone(cut, cnn_config(arch)['cut'])
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i, o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int):
        return nn.Sequential(*(list(model.children())[:cut]))
    elif isinstance(cut, Callable):
        return cut(model)
    else:
        pass


if __name__ == '__main__':

    from fastai.vision.learner import _squeezenet_split

    # arch = torchvision.models.densenet121
    arch = pretrainedmodels.resnet34
    # arch = pretrainedmodels.se_resnext50_32x4d

    body = create_body_local(
        arch=arch,
        pretrained=True,
        cut=None)  # extract encoder part from encoder network

    # body = create_body(
    #     arch=arch,
    #     pretrained=True,
    #     cut=None)  # extract encoder part from encoder network

    body = body if len(body) > 1 else body[0]

    model = DynamicUnetAttention_Hcolumns(
        encoder=body,
        n_classes=2,
        imsize=(256, 256),
        self_attention=True
    )

    summary(model, (3, 256, 256), device='cpu')