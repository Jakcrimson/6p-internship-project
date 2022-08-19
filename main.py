"""
@author - Pierre Lague & Javiera Castillo
"""
from options import TrainOptions
from train import TrainModel
from test import TestModel

import torch

options = TrainOptions().parser.parse_args()

torch.cuda.set_device(options.gpu_id)

if options.train :
    model = TrainModel(options)
    model.trainProcess()

else:
    model = TestModel(options)
    model.testProcess()