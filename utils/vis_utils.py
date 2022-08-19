
import torch
import torchvision

import numpy as np

from ipdb import set_trace as st

_palette = {
        0: (219, 95, 87),
        1: (219, 151, 87),
        2: (219, 208, 87),
        3: (173,219,87),
        4: (117, 219, 87),
        5: (123, 196., 123),
        6: (88, 177,  88),
        7: (212, 246, 212),
        8: (176, 226, 176),
        9: (0, 128, 0),
        10: (88, 176, 167),
        11: (153,  93,  19),
        12: (87, 155, 219),
        13: (0, 98, 255),
        14: (255, 255, 255)
        }


sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
plot = lambda p, x: torchvision.utils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))
visualize = lambda x : torchvision.utils.make_grid(torch.clamp(x, -1, 1), normalize=True, nrow=sqrt(x.size(0)))

def visualize_labels(preds, palette=_palette):
    b, w, h = preds.shape
    vis = np.zeros((b,3, w,h))
    for i, pred in enumerate(preds):
        vis[i] = convert_to_color(pred, palette)
    vis = torch.from_numpy(vis)
    return torchvision.utils.make_grid(vis, normalize=True, value_range=(0,255), nrow=sqrt(vis.size(0)))

def convert_to_color(arr_2d, palette):
    """ Numeric labels to RGB-color encoding """
    if torch.is_tensor(arr_2d):
        arr_2d = arr_2d.cpu().numpy()

    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.int)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    
    return arr_3d.astype('uint8').transpose(2,0,1)
