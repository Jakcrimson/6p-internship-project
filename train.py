#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors - Javiera Castillo & Pierre Lague
@version - 1.0
@project - SixP
"""

## Project libs
from datasets import sixp_dataset
from utils.utils import get_logger
from utils.train_utils import TBLog, simple_accuracy
from utils.vis_utils import visualize, visualize_labels

# DL libs
import torch.utils.data
import torch.nn.functional as F
from torch import nn
import torch.optim as optim 
import segmentation_models_pytorch as smp

# Utils libs
import numpy as np
import os


"""
Class called when the options --train is specified in the command line
"""
class TrainModel():

    def __init__(self, options):
        self.options = options
        self.checkpoints_dir = self.create_checkpoints_dir(options)
        self.tb_log = TBLog(self.checkpoints_dir, 'tensorboard')
        self.logger = get_logger(options.name, self.checkpoints_dir, level='INFO')
        self.print_fn = self.logger.info
        self.iter = 0 # TO DO: Find a better way to count iterations.

    """
    Function called to train a model. It defines the (network, backbone), the optimizer and the datasets.
    """
    def trainProcess(self):

        # Define the network to be used
        net = self._get_network()
        net.cuda()
        
        # Define the optimizer
        optimizer = self._get_optimizer(net)
        
        # Define the dataset
        dset = self._get_dataset()
        if not self.options.val_root is None :
            valset = self._get_valset()
            valLoader = torch.utils.data.DataLoader(valset, batch_size=self.options.batch_size)
        else:
            valLoader = None
        dataloader = torch.utils.data.DataLoader(dset, batch_size=self.options.batch_size)
            
        print('\n Training ... \n')

        ## If cached training needs to be loaded
        if self.options.start_epoch > 1:
            checkpoint_path = os.path.join(self.options.checkpoints_dir, self.options.name, 'latest.pth.tar')

            net.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)['model_state_dict'])
            optimizer.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)['optimizer_state_dict'])   

        ## Looping through the epochs and computing metrics.
        for epoch in range(self.options.start_epoch, self.options.nEpochs+1):
            # Compute a training epoch
            self.trainEpoch(valLoader, dataloader, net, optimizer, epoch, self.iter)
    
            if epoch % self.options.save_freq == 0:
                save_name = 'epoch_{:04}.pth'.format(epoch)
                self._save_checkpoint(save_name, self.checkpoints_dir, net, optimizer, epoch) 

    """
    Function called to train the model during 1 epoch.
    @params :
    - valLoader : valididation dataLoader
    - dataLoader : train dataLoader
    - net : the model
    - optimizer : the optimizer
    - epoch : which epoch to train (range(1, nEpochs))
    - iter : the number of epochs that have been trained yet
    """
    def trainEpoch(self, valLoader, dataloader, net, optimizer, epoch, iter):
        start_epoch = torch.cuda.Event(enable_timing=True)
        end_epoch = torch.cuda.Event(enable_timing=True)
        tb_dict = {}
        net.train()
        start_epoch.record()

        #TRAINING
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = net(data)
            
            y_target = target.data.cpu().numpy()
            
            #defining weights for weighted cross entropy
            weight_undef = np.array([0])
            
            # Get class weights (inverse frequency) from training labels - weights are dynamic with the batch.
            labels = np.concatenate(y_target, 0)  # labels.shape = (n_samples, int(self.options.n_classes).
            classes = labels[:, 0].astype(np.int) 
            weights = np.bincount(classes, minlength=int(self.options.n_classes))  # occurences per class 
            weights[weights == 0] = 1  # replace empty bins with 1 
            weights = 1 / weights  # number of targets per class 
            weights /= weights.sum()  # normalize 
            
            weights[0] = 0.0 #we don't want the undef class
            
            weighted_classes = torch.tensor(weights, dtype=torch.float).cuda()

            #computing cross entropy loss
            cross_entropy_loss = F.cross_entropy(output, target, weight=weighted_classes)
            cross_entropy_loss.backward()

            #computing stats in order to compute the metrics
            tp, fp, fn, tn = smp.metrics.get_stats((torch.argmax(output-1, dim=1)).type(torch.long), (target-1), mode='multiclass', num_classes=self.options.n_classes-1, ignore_index=-1)                     

            optimizer.step()

            #inserting the metrics in the tensorboard            
            tb_dict['IoU'] = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            tb_dict['F1 Score (DICE)'] = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            tb_dict['Recall'] = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
            tb_dict['train/sup_loss'] = cross_entropy_loss.detach() 
            tb_dict['lr'] = optimizer.param_groups[0]['lr']
            tb_dict['train/accuracy'] = simple_accuracy(torch.argmax(output, dim=1), target)

            #inserting the images in the tensorboard
            tb_img_dict = {}

            if (self.options.print_info_freq > 0) & (self.iter % self.options.print_info_freq == 0):
                self.print_fn(tb_dict)

            if not self.tb_log is None:
                self.tb_log.update(tb_dict, self.iter)

                if (self.options.display_freq > 0) & (self.iter % self.options.display_freq == 0):
                    img_dict = self._get_images(data, None, target, output)
                    tb_img_dict.update(img_dict)
                    self.tb_log.update_imgs(tb_img_dict, it=self.iter)
            self.iter +=1
        end_epoch.record()


        if not self.options.val_root is None and self.iter%10 == 0:
            #VALIDATION
            with torch.no_grad():
                net.eval()
                num_correct = 0
                num_pixels = 0

                for batch_idx, (x, y) in enumerate(valLoader):
                    x = x.cuda()
                    y = y.cuda()

                    #computing output and predictions on the validation set 
                    output = net(x)
                    preds = torch.argmax(output, dim=1)
                    
                    #defining weights for weighted cross entropy
                    weight_undef = np.array([0])
                    weights_other = np.ones(self.options.n_classes - 1)
                    weights_cross_entropy = np.concatenate((weight_undef, weights_other), axis=None)
                    weighted_classes = torch.tensor(weights_cross_entropy, dtype=torch.float).cuda()
                    #computing the cross entropy loss
                    cross_entropy_loss = F.cross_entropy(output, y, weight=weighted_classes)

                    ##compute metrics
                    num_correct += (preds == y).sum()
                    num_pixels += torch.numel(preds)
                    
                    tb_dict['validation/sup_loss'] = cross_entropy_loss.detach() 
                    tb_dict['validation/accuracy'] = simple_accuracy(torch.argmax(output, dim=1), y)

                    if (self.options.print_info_freq > 0) & (self.iter % self.options.print_info_freq == 0):
                        self.print_fn(tb_dict)

                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.iter)
            net.train()
            torch.cuda.synchronize()

            print(
                f"##VALIDATION PROCESS##\nGot {num_correct}/{num_pixels} pixels correct with Accuracy -> {num_correct/num_pixels*100:.2f}%"
            )


        print('Time per epoch : ', start_epoch.elapsed_time(end_epoch)/1000)
         

    """
    Creates the directory the store the checkpoints.
    example : ./checkpoints/unet/resnet50/
    """
    def create_checkpoints_dir(self, options):
        checkpoint_path = os.path.join(options.checkpoints_dir, options.name, options.encoder) 
        os.makedirs(checkpoint_path, exist_ok=True)
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
        elif net_arch == 'dlv3plus':
            return smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=out_channels_sup)
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
    Returns the validation dataset. Since we don't have enough data, we're forced to use the test set as a validation set.
    """
    def _get_valset(self):
        # dataset to be used [SixP]
        dataset = self.options.dataset

        if dataset == 'sixp':
            lb_dataset = sixp_dataset.sixP(self.options.val_root, normalize=True)
            return lb_dataset        
        else:
            raise NameError("Dataset must be SixP. Others need to be implemented.")

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
    
    """
    Saves the current state of the model to the checkpoints dir.
    """
    def _save_checkpoint(self, save_name, save_path, model, optimizer, epoch):
        save_filename = os.path.join(save_path, save_name)
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter': self.iter,
                    'epoch': epoch}, save_filename)
        self.print_fn(f"model saved: {save_filename}")
        pass