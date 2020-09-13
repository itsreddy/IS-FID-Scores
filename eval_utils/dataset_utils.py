import os, sys, torch, pickle
from tqdm.notebook import tqdm_notebook as tq
import numpy as np
import random
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset
from torch.autograd import Variable

base_path = "/content/drive/My Drive/UC Davis Synthetic Data/Prashanth's/cifar/evaluation/"
sys.path.append(base_path)
from eval_utils.misc_utils import save_obj

class GenerateImages:
    def gen_image_batches(g, opt, save_path, n=50000, batch=2000):
        '''
            Generate and save large batches of images in pickled format
            input: generator, options, savepath, number of images, batch size
        '''
        Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor
        b = 1
        for i in tq(range(0, n, batch)):
            z = Variable(Tensor(np.random.normal(0, 1, (batch, opt.latent_dim))))
            g_imgs = g(z).detach().cpu().numpy()
            save_obj(g_imgs, save_path + 'batch_{}'.format(b))
            b += 1
        print("images saved at: ", save_path)

class NewDataset(VisionDataset):
    '''
        pytorch dataset object to access large batches of images effeciently
        input: transform
    '''

    def __init__(self, root, transform=None):
        super(NewDataset, self).__init__(None, transform=transform,
                                      target_transform=None)

        self.base_folder = root
        self.file_list = os.listdir(self.base_folder)
        self.data = []

        # now load the picked numpy arrays
        for file_name in self.file_list:
            file_path = os.path.join(self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f)
                self.data.append(entry)

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        
    def __getitem__(self, index):
        img = self.data[index]

        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data)