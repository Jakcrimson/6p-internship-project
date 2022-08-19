#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author - Pierre Lague
@version - 1.0
@project - SixP
"""

## Project libs
from datasets import sixp_dataset
from utils.utils import get_logger
from utils.train_utils import TBLog, simple_accuracy
from utils.vis_utils import visualize, visualize_labels
from utils.utils import count_parameters, load_checkpoint

# DL libs
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim 
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Utils libs
import numpy as np
from skimage import measure
from sklearn import metrics
import os
import matplotlib.pyplot as plt
import numpy as np


"""
Class called when the train option isn't specified in the command line.
It tests a model.
"""
class TestModel():

    def __init__(self, options):
        self.options = options
        self.iou = 0
        self.dice = 0
        self.precision = 0
        self.recall = 0

        self.checkpoints_dir = self.get_checkpoints_dir(options)
        self.tb_log = TBLog(self.checkpoints_dir, 'tensorboard_test')
        self.logger = get_logger(options.name, self.checkpoints_dir, level='INFO')
        self.print_fn = self.logger.info
        self.iter = 0 # TO DO: Find a better way to count iterations.

    """
    Similar to the trainProcess function, this will loop in the test set and compute metrics in order to evaluate our model
    """
    def testProcess(self):
        # Define the network to be used
        net = self._get_network()
        net.cuda()
        
        # Define the optimizer
        optimizer = self._get_optimizer(net)
        
        # Define the dataset
        dset = self._get_dataset()
        dataloader = torch.utils.data.DataLoader(dset, batch_size=self.options.batch_size)

        #Loading the correct checkpoint : since there's been overtfitting after epoch 30 we load the epoch_0030.pth checkpoint.
        checkpoint_path = os.path.join(self.options.checkpoints_dir, self.options.name, self.options.encoder, 'epoch_0100.pth')
        checkpoint = torch.load(checkpoint_path)

        #Loading the model and the optimizer
        net.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)['model'])
        optimizer.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)['optimizer'])   

        print('\n Testing ... \n')
        print('\n Number of parameters :', count_parameters(net))
        folder_path = os.path.join(self.options.checkpoints_dir, self.options.name)
        
        #calling the function that computes all metrics and evaluates the model
        self.check_accuracy(dataloader, net, folder_path)
    
    """
    Returs the training dataset (there's only one dataset which is SIXP but for methodology means its an important function)
    """
    def _get_dataset(self):
        # dataset to be used [SixP]
        dataset = self.options.dataset

        if dataset == 'sixp':
            lb_dataset = sixp_dataset.sixP(self.options.data_root, normalize=True)
            return lb_dataset        
        else:
            raise NameError("Dataset must be SixP. Others need to be implemented.")

    """
    Returns the checkpoint directory.
    """
    def get_checkpoints_dir(self, options):
        checkpoint_path = os.path.join(options.checkpoints_dir, options.name, options.encoder) 
        return checkpoint_path

    """
    Returns the network built with a specified encoder via the SMP lib.
    """
    def _get_network(self):
        net_arch = self.options.net_architecture
        in_channels = self.options.input_nc
        out_channels_sup = self.options.n_classes
        encoder_name = self.options.encoder

        if net_arch == 'unet':
            return smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=out_channels_sup)
        elif net_arch == 'unetPlusPlus':
            return smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=out_channels_sup)
        elif net_arch == 'manet':
            return smp.MAnet(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=out_channels_sup)
        elif net_arch == 'linknet':
            return smp.Linknet(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=out_channels_sup)
        elif net_arch == 'fpn':
            return smp.FPN(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=out_channels_sup)
        elif net_arch == 'pspnet':
            return smp.PSPNet(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=out_channels_sup)
        elif net_arch == 'dlv3':
            return smp.DeepLabV3(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=out_channels_sup)
        else:
            raise NotImplementedError("Your input network is not supported (yet)")

    """
    Returns the optimizer
    """
    def _get_optimizer(self, net):
        # For now we will only use Adam optimizer
        # TO DO: Add other options to choose the optimizer
        return optim.Adam(net.parameters(), lr=self.options.lr)

    """
    input : m -> ground truth (w, h image with values in {0, 1, ..., K} K the number of classes)
            p -> prediction by the model of an image
    -> need to define lambda and sigma
    """
    def compute_metrics(self, m, p, n_classes, thresh):
        TP_r = np.zeros(n_classes)
        FP_r = np.zeros(n_classes)
        FN_r = np.zeros(n_classes)
        
        for i in np.unique(m): #for each plant class do the same
          p = (p == i) # get predicted binary mask for the i-th class
          m = (m == i) # get binary ground truth for the i-th class
          cc = measure.label(m) # measure and label all connected regions of m

          for j in np.unique(cc):
            c = (cc==j)
            c[c > 1] = 1
            intersection = np.multiply(c, p)

            if sum(sum(intersection))/sum(sum(c)) > thresh  : # we suppose intersection & C are binary masks with Os and 1s . ( so that the sum represent number of píxels with value = 1)
                TP_r[i] = TP_r[i] + 1
                #print(i)
            else:          
                FN_r[i] = FN_r[i] + 1
                #print(i)

          b = np.subtract(p, (np.multiply(m, p)), dtype=np.int32) # get any pixels predicted as i , but not in the groundtruth
          
          ccb = measure.label(b) # get all connected regions of prediction for the i-th class.
          
          for j in np.unique(ccb):
            c = (ccb == j)
            if sum(sum(c)) >= 10000: # make sure that c is binary with values in { 0,1 } so that FP + = 1 . sum (c) is the number of píxels with value = 1 
              FP_r[i] = FP_r[i] + 1

        return TP_r, FP_r, FN_r


    """
    Function that computes the metrics for a given model
    """
    def check_accuracy(self, loader, model, folder, device="cuda"):
        num_correct = 0
        num_pixels = 0
        model.eval()
        tb_dict = {}
        TP = 0
        FP = 0
        FN = 0
        precisions1= []
        recalls1 = []
        np.seterr(invalid='ignore')
        thresholds = np.arange(start=0.1, stop=0.7, step=0.05)
        for thresh in thresholds:
          #evaluating the model
        
          with torch.no_grad():
              for batch_idx, (x, y) in enumerate(loader):
                  sample_fname = loader.dataset._load_files()
                  x = x.cuda()
                  y = y.cuda()
                  
                  output = model(x)
                  preds = torch.argmax(output, dim=1)

                  y_true = y.data.cpu().numpy()
                  y_pred = (torch.argmax(output, dim=1)).data.cpu().numpy()

                  y_pred = y_pred.squeeze(0)
                  y_true = y_true.squeeze(0)
                                
    ####################################################################################################    

                  tp, fp, fn = self.compute_metrics(y_true, y_pred, self.options.n_classes, thresh)
                  TP+=tp
                  FP+=fp
                  FN+=fn
                    
    ####################################################################################################
                  num_correct += (preds == y).sum()
                  num_pixels += torch.numel(preds)
              precisions1.append(TP / (TP + FP))
              recalls1.append(TP / (TP + FN))
          
        AUC = []
        for i in range(len(precisions1)):
          for j in range(int(self.options.n_classes)):
            precisions_thresholds = []
            recall_thresholds = []
            precisions_thresholds.append(precisions1[i][j])
            recall_thresholds.append(recalls1[i][j])
        
            # computing the average precision
            precisions_thresholds = [precisions_thresholds[0]] + precisions_thresholds.copy() + [0]
            recall_thresholds = [0] + recall_thresholds.copy() + [recall_thresholds[-1]]

            #plotting the average precision
            assert len(precisions_thresholds) == len(recall_thresholds)
            precisions_thresholds.sort()
            recall_thresholds.sort()
            # Using AUC as AP
            auc = metrics.auc(x=recall_thresholds, y=precisions_thresholds)
            AUC.append(auc)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.axis("square")
            ax.plot(recall_thresholds[1:-1], precisions_thresholds[1:-1], "-bo", label=None, clip_on=False)
            ax.plot(recall_thresholds[0:2], precisions_thresholds[0:2], "-ro", label=None, clip_on=False)
            ax.plot(recall_thresholds[-2:], precisions_thresholds[-2:], "-go", label=None, clip_on=False)
            ax.set_xlim([0, 1.0])
            ax.set_ylim([0, 1.0])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
            ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
            ax.set_title(f"AP = {auc}", fontsize=14, fontweight="bold")
            ax.fill_between(recall_thresholds[1:-1], precisions_thresholds[1:-1], color="skyblue")
            ax.fill_between(recall_thresholds[0:2], precisions_thresholds[0:2], color="salmon")
            ax.fill_between(recall_thresholds[-2:-1], precisions_thresholds[-2:-1], color="lime")
            fig.savefig("AP_CLASS_"+str(j)+".png", format="png", dpi=600, bbox_inches="tight")

            precisions_thresholds = []
            recall_thresholds = []
        print(f"Mean Average Precision = {sum(AUC)/len(AUC)}")

    """ 
    Retrieves images from the sets in order to put them in the tensorboard
    """    
    def _get_images(self, x_lb, x_ulb, y_lb, preds):
        img_dict = {}
        img_dict['images'] = visualize(x_lb)
        if x_ulb is not None:
            img_dict['ulb_imgs'] = visualize(x_ulb)
        img_dict['ground truth'] = visualize_labels(y_lb)
        prediction = torch.argmax(preds, dim=1)
        img_dict['prediction'] = visualize_labels(prediction)
        return img_dict

  
    