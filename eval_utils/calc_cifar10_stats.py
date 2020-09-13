from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from eval_utils.score_utils import ScoreModel
import numpy as np

def calc_cifar10_stats(save_path, select_classes=None):
    '''
        calculate IS, mu, sigma for CIFAR10 dataset
        input: save_path
        savepath is usually in the form of: base_path + '/data/cifar10'
    '''
    class IgnoreLabelDataset(Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    cifar = CIFAR10(root=save_path, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
    
    if select_classes:
        # pick only classes from list
        select = lambda i: True if i in set(select_classes) else False
        idx = [select(i) for i in cifar.targets]
        cifar.data = cifar.data[idx]
    
    # IgnoreLabelDataset(cifar)
    is_fid_model = ScoreModel(mode=2, cuda=True)
    is_mean, is_std, _, mu, sigma = is_fid_model.get_score_dataset(IgnoreLabelDataset(cifar),
                                                                    n_split=10, return_stats=True)
    
    np.savez_compressed(save_path, mu=mu, sigma=sigma)
    print(is_mean, is_std)
    print('saved stats: ', mu, sigma)