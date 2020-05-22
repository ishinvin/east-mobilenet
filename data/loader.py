from data.dataset import ICDAR
from torch.utils.data import DataLoader
from data.utils import collate_fn


class ICDARDataLoader:

    def __init__(self, config):
        self.config = config
        self.batch_size = config['data_loader']['batch_size']
        self.shuffle = config['data_loader']['shuffle']
        self.num_workers = config['data_loader']['workers']
        data_root = config['data_loader']['data_dir']
        input_size = config['data_loader']['input_size']
        self.icdar_dataset = ICDAR(data_root, input_size)

    def train(self):
        return DataLoader(self.icdar_dataset, num_workers=self.num_workers, batch_size=self.batch_size,
                          shuffle=self.shuffle, collate_fn=collate_fn)
