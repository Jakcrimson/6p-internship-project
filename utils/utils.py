import os
import time
from torch.utils.tensorboard import SummaryWriter
import logging
import torch
import torchvision
import segmentation_models_pytorch as smp
from utils.train_utils import TBLog, simple_accuracy


def setattr_cls_from_kwargs(cls, kwargs):
    #if default values are in the cls,
    #overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls,key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])

        
def test_setattr_cls_from_kwargs():
    class _test_cls:
        def __init__(self):
            self.a = 1
            self.b = 'hello'
    test_cls = _test_cls()
    config = {'a': 3, 'b': 'change_hello', 'c':5}
    setattr_cls_from_kwargs(test_cls, config)
    for key in config.keys():
        print(f"{key}:\t {getattr(test_cls, key)}")

    
def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)
    
    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    
    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(options, loader, model, folder, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            preds = torch.argmax(output, dim=1)
            preds = (preds > 0.5).float()  

            #stats
            tp, fp, fn, tn = smp.metrics.get_stats(preds, y, mode='multilabel', threshold=0.5)                      
            
            #compute metrics
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            
            #put them in the tb logger
            tb_dict = {}
            tb_dict['IoU'] = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            tb_dict['F1 Score (DICE)'] = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            tb_dict['Recall'] = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
            tb_dict['test/accuracy'] = simple_accuracy(torch.argmax(output, dim=1), y)

            tb_img_dict = {}

            if (options.print_info_freq > 0) & (self.iter % self.options.print_info_freq == 0):
                self.print_fn(tb_dict)

            if not self.tb_log is None:
                self.tb_log.update(tb_dict, self.iter)

                if (self.options.display_freq > 0) & (self.iter % self.options.display_freq == 0):
                    img_dict = self._get_images(data, None, target, output)
                    tb_img_dict.update(img_dict)
                    self.tb_log.update_imgs(tb_img_dict, it=self.iter)
            
            torch.cuda.synchronize()

            

    print(
        f"Got {num_correct}/{num_pixels} pixels correct with Accuracy -> {num_correct/num_pixels*100:.2f}%"
    )
    print(f"Dice score : {dice/len(loader)}")
    print(f"IoU : {iou/len(loader)}")
    
    
    model.train()

    model.eval()
    

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        #torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()