import torch
from torch.utils.data import Dataset

class HodaDataset(Dataset):
    def __init__(self,train=True,M=60000):
        # data loading
        if train:
            self.x = torch.load( '../data/HODA/hoda_data_train.pt')[:M,:,:,:]
            self.y = torch.load( '../data/HODA/hoda_labels_train.pt')[:M]
        else:
            self.x = torch.load( '../data/HODA/hoda_data_test.pt')
            self.y = torch.load( '../data/HODA/hoda_labels_test.pt')
        self.n_samples = self.x.shape[0]
    def __getitem__(self,index):
        # dataset[index]
        return self.x[index], self.y[index]
    def __len__(self):
        # len(dataset)
        return self.n_samples

