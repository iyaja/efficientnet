from fastai2.vision.all import *
from fastai2.callback.wandb import *
import pandas as pd
import scipy.io as sio
import wandb
import geffnet
from fastai2.distributed import *

hyperparameter_defaults = dict(
    efficientnet_b = 3,
    )


run = wandb.init(config=hyperparameter_defaults, project='efficientnet')

path = untar_data(URLs.IMAGENETTE)

dls = ImageDataLoaders.from_folder(path, valid='val', bs=32, item_tfms=Resize(224))

def efficientnet(variant, pretrained=False):
  if variant is None:
    print("please specify an EfficientNet variant (b0-b7). Using b0 as default.")
    model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=pretrained)
    return model
    # return geffnet.efficientnet_b0(pretrained=pretrained, drop_rate=0.25, drop_connect_rate=0.2, as_sequential=True)
  if variant==0:
    return geffnet.efficientnet_b0(pretrained=pretrained, as_sequential=True)
  if variant==1:
    return geffnet.efficientnet_b1(pretrained=pretrained, as_sequential=True)
  elif variant==2:
    return geffnet.efficientnet_b2(pretrained=pretrained, as_sequential=True)
  elif variant==3:
    return geffnet.efficientnet_b3(pretrained=pretrained, as_sequential=True)
  elif variant==4:
    return geffnet.efficientnet_b4(pretrained=pretrained, as_sequential=True)
  elif variant==5:
    return geffnet.efficientnet_b5(pretrained=pretrained, as_sequential=True)
  elif variant==6:
    return geffnet.efficientnet_b6(pretrained=pretrained, as_sequential=True)
  elif variant==7:
    return geffnet.efficientnet_b7(pretrained=pretrained, as_sequential=True)

b = run.config.efficientnet_b

learn = cnn_learner(dls, partial(efficientnet, b), pretrained=False, metrics=accuracy).to_parallel()
learn.fit_one_cycle(50, cbs=WandbCallback(log_preds=False))
