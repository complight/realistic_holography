import glob
import torch
from odak.learn.tools import load_image

def load(fn,device):
    target = load_image(fn).to(device).double()
    if len(target.shape) > 2:
        if target.shape[2] > 2:
            target = target[:,:,0:3]
    if len(target.shape) > 2:
        target = torch.mean(target,2)
    if target.max() > 1.: 
        target = target/255.
    return target.float()

class DatasetFromFolder():
    def __init__(self,input_directory,output_directory,device,key='.png'):
        self.device           = device
        self.key              = key
        self.input_directory  = input_directory
        self.output_directory = output_directory
        self.input_filenames  = sorted(glob.glob(input_directory+'/**/*{}'.format(self.key),recursive=True))
        self.output_filenames = sorted(glob.glob(output_directory+'/**/*{}'.format(self.key),recursive=True))

    def __getitem__(self, index):
        input_image  = load(self.input_filenames[index],self.device)
        output_image = load(self.output_filenames[index],self.device)
        return input_image,output_image

    def __len__(self):
        return len(self.input_filenames)

