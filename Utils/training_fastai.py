from fastai.vision import *
from fastai.vision.learner import has_pool_type

from fastai.vision.learner import cnn_config, num_features_model, create_head
from fastai.vision.models import DynamicUnet
from Utils.acc_grad_learner import AccGradLearner
import torch


def create_body_local(arch: Callable,
                      is_se_resnext=False,
                      pretrained=None,
                      cut=None):

    "Cut off the body of a typically pretrained `model` at `cut` (int) or cut the model as specified by `cut(model)` (function)."

    if pretrained is None:
        if is_se_resnext:
            model = arch(num_classes=1000, pretrained='imagenet')
        else:
            model = arch(pretrained=True)
    else:
        model = create_cnn_model(arch, is_se_resnext=is_se_resnext, nc=2)
        model.load_state_dict(torch.load(pretrained)['model'])

        return model[:1]

    cut = ifnone(cut, cnn_config(arch)['cut'])

    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i, o in reversed(ll) if has_pool_type(o))

    if isinstance(cut, int):
        return nn.Sequential(*(list(model.children())[:cut]))
    elif isinstance(cut, Callable):
        return cut(model)
    else:
        print('Wrong inputs!')
        return


def unet_learner_e(
        data: DataBunch,
        arch: Callable,
        imsize: Tuple,
        unet_model: Callable = None,
        is_se_resnext=False,
        pretrained=None,
        blur_final: bool = True,
        norm_type: Optional[NormType] = NormType,
        split_on: Optional[SplitFuncOrIdxList] = None,
        blur: bool = False,
        self_attention: bool = False,
        y_range: Optional[Tuple[float, float]] = None,
        last_cross: bool = True,
        bottle: bool = False,
        cut: Union[int, Callable] = None,
        n_classes=2,
        device=torch.device('cuda'),
        **learn_kwargs: Any) -> Learner:

    "Build Unet learner from `data` and `arch`."

    meta = cnn_config(arch)  # get metadata of encoder network

    body = create_body_local(
        arch=arch,
        is_se_resnext=is_se_resnext,
        pretrained=pretrained,
        cut=cut
    )

    body = body if len(body) > 1 else body[0]

    if unet_model is None:
        model = to_device(DynamicUnet(body, n_classes=n_classes, blur=blur, blur_final=blur_final,
                                      self_attention=self_attention, y_range=y_range, norm_type=norm_type,
                                      last_cross=last_cross, bottle=bottle), device)
    else:

        model = to_device(unet_model(body, n_classes=n_classes, imsize=imsize, blur=blur, blur_final=blur_final,
                                     self_attention=self_attention, y_range=y_range, norm_type=norm_type,
                                     last_cross=last_cross, bottle=bottle), device)

    # learn = Learner(data, model, **learn_kwargs)
    learn = AccGradLearner(data, model, **learn_kwargs)

    learn.split(ifnone(split_on, meta['split']))

    learn.freeze()

    apply_init(model[2], nn.init.kaiming_normal_)

    return learn


def create_cnn_model(base_arch: Callable, nc: int, cut: Union[int, Callable] = None,
                     pretrained=None, is_se_resnext=False,
                     lin_ftrs: Optional[Collection[int]] = None, ps: Floats = 0.5,
                     custom_head: Optional[nn.Module] = None,
                     split_on: Optional[SplitFuncOrIdxList] = None, bn_final: bool = False,
                     concat_pool: bool = True):

    "Create custom convnet architecture"
    body = create_body_local(
        arch=base_arch,
        is_se_resnext=is_se_resnext,
        pretrained=pretrained,
        cut=cut
    )

    if custom_head is None:
        nf = num_features_model(nn.Sequential(*body.children())) * (2 if concat_pool else 1)
        head = create_head(nf, nc, lin_ftrs, ps=ps, concat_pool=concat_pool, bn_final=bn_final)
    else:
        head = custom_head

    return nn.Sequential(body, head)


def cnn_learner_e(data:DataBunch,
                  base_arch:Callable,
                  pretrained=None,
                  is_se_resnext=False,
                  cut:Union[int, Callable]=None,
                  lin_ftrs:Optional[Collection[int]]=None,
                  ps:Floats=0.5,
                  custom_head:Optional[nn.Module]=None,
                  split_on:Optional[SplitFuncOrIdxList]=None,
                  bn_final:bool=False,
                  init=nn.init.kaiming_normal_,
                  concat_pool:bool=True,
                  device=torch.device('cuda'),
                  **kwargs:Any)->Learner:

    "Build convnet style learner."
    meta = cnn_config(base_arch)

    model = to_device(create_cnn_model(base_arch, nc=data.c, cut=cut,
                                       pretrained=pretrained, is_se_resnext=is_se_resnext,
                                       lin_ftrs=lin_ftrs, ps=ps, custom_head=custom_head,
                                       bn_final=bn_final, concat_pool=concat_pool), device)

    learn = AccGradLearner(data, model, **kwargs)

    learn.split(split_on or meta['split'])

    learn.freeze()

    if init:
        apply_init(model[1], init)

    return learn
