"""
@author - Pierre Lague & Javiera Castillo
"""
import argparse

class TrainOptions():
    # TO DO: Add choices argument to options that support limited values.

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Options to train our neural networks for semantic segmentation')
        # basic parameters
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--train', action='store_true', help='indicates training phase. Il False, testing mode is on, provide test data_root accordingly')
        self.parser.add_argument('--data_root', type=str)
        self.parser.add_argument('--val_root', type=str)

        # model parameters
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        self.parser.add_argument('--n_classes', type=int, required=True, help='number of output channels for segmentation')
        self.parser.add_argument('--net_architecture', type=str, default='unet', help='specify architecture [unet, unetPlusPlus, manet, linknet, fpn, pspnet, dlv3]')
        self.parser.add_argument('--encoder', type=str, default='resnet50', help='specify the encoder to use in the model [https://smp.readthedocs.io/en/latest/encoders.html]')
       
        # dataset parameters
        self.parser.add_argument('--dataset', type=str, required=True, help='dataset to be used [sixp]')
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--window_size', type=int, default=[256,256], nargs=2, help='patch size for training')


        # training parameters
        self.parser.add_argument('--start_epoch', type=int, default=1, help='starting epoch for training. If > 1 it will load existing cached training')
        self.parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to be used')

        # visualization options
        self.parser.add_argument('--print_info_freq', type=int, default=10)
        self.parser.add_argument('--display_freq', type=int, default=15)


        # network saving and loading parameters
        self.parser.add_argument('--save_freq', type=int, default=10, help='frequency of saving the latest results (in terms of epoch)')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')

        # testing options
        self.parser.add_argument('--test_stride', type=int, default=256, help='stride for sliding window during test')
        self.parser.add_argument('--test_pth', type=str, default='latest.pth.tar', help='checkpoint to load and test')

       