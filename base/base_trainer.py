import os
import math
import torch
import logging

from tensorboardX import SummaryWriter
from utils.util import make_dir


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, loss, resume, config, train_logger=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.loss = loss
        self.name = config['name']
        self.epochs = config['trainer']['epochs']
        self.save_freq = config['trainer']['save_freq']
        self.verbosity = config['trainer']['verbosity']
        self.summary_writer = SummaryWriter()

        # check cuda available
        if torch.cuda.is_available():
            if config['cuda']:
                self.with_cuda = True
                self.gpus = {i: item for i, item in enumerate(self.config['gpus'])}
                device = 'cuda'
                if torch.cuda.device_count() > 1 and len(self.gpus) > 1:
                    self.model.parallelize()
                torch.cuda.empty_cache()
            else:
                self.with_cuda = False
                device = 'cpu'
        else:
            self.logger.warning('Warning: There\'s no CUDA support on this machine, training is performed on CPU.')
            self.with_cuda = False
            device = 'cpu'

        self.device = torch.device(device)
        self.model.to(self.device)

        # log
        self.logger.debug('Model is initialized.')
        self._log_memory_useage()
        self.train_logger = train_logger

        # optimizer
        self.optimizer = self.model.optimize(config['optimizer_type'], config['optimizer'])

        # train monitor
        self.monitor = config['trainer']['monitor']
        self.monitor_mode = config['trainer']['monitor_mode']
        assert self.monitor_mode == 'min' or self.monitor_mode == 'max'
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf

        # checkpoint path
        self.start_epoch = 1
        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], self.name)
        make_dir(self.checkpoint_dir)

        if resume:
            self._resume_checkpoint(resume)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        print('Total epochs: {}'.format(self.epochs))
        for epoch in range(self.start_epoch, self.epochs + 1):
            try:
                result = self._train_epoch(epoch)
            except torch.cuda.CudaError:
                self._log_memory_useage()

            log = {'epoch': epoch}
            for key, value in result.items():
                log[key] = value

            # log info
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))

            # save checkpoints
            if (self.monitor_mode == 'min' and log[self.monitor] < self.monitor_best) \
                    or (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
                self.monitor_best = log[self.monitor]
                self._save_checkpoint(epoch, log, save_best=True)

            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch, log)

            self.summary_writer.add_scalars('HMEAN', {'hmean': result['hmean']}, epoch)
            self.summary_writer.add_scalars('LOSS', {'train_loss': result['loss']}, epoch)

        self.summary_writer.close()

    def _log_memory_useage(self):
        if not self.with_cuda:
            return

        template = """Memory Usage: \n{}"""
        usage = []
        for deviceID, device in self.gpus.items():
            deviceID = int(deviceID)
            allocated = torch.cuda.memory_allocated(deviceID) / (1024 * 1024)
            cached = torch.cuda.memory_cached(deviceID) / (1024 * 1024)
            usage.append('    CUDA: {}  Allocated: {} MB Cached: {} MB \n'.format(device, allocated, cached))

        content = ''.join(usage)
        content = template.format(content)

        self.logger.debug(content)

    def _save_checkpoint(self, epoch, log, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        arch = type(self.model).__name__
        if save_best:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict()
            }
            filename = os.path.join(self.checkpoint_dir, 'model_best.pth.tar')
            torch.save(state, filename)
            self.logger.info("Saving current best: {} ...".format('model_best.pth.tar'))
        else:
            state = {
                'arch': arch,
                'epoch': epoch,
                'logger': self.train_logger,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'monitor_best': self.monitor_best
            }
            filename = os.path.join(self.checkpoint_dir,
                                    'checkpoint-epoch{:03d}-loss-{:.4f}.pth.tar'.format(epoch, log['loss']))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.with_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(torch.device('cuda'))
        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
