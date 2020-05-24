import torch
import pathlib
from base.base_trainer import BaseTrainer
from eval import main_evaluate
from utils.util import make_dir


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """

    def __init__(self, model, loss, resume, config, data_loader, train_logger=None):
        super(Trainer, self).__init__(model, loss, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.with_gpu = True if torch.cuda.is_available() else False
        self.root_dataset = config['data_loader']['data_dir']

    def _to_tensor(self, *tensors):
        t = []
        for __tensors in tensors:
            t.append(__tensors.to(self.device))
        return t

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()

        total_loss = 0
        for batch_idx, gt in enumerate(self.data_loader):
            try:
                image_paths, img, score_map, geo_map, training_mask = gt
                img, score_map, geo_map, training_mask = self._to_tensor(img, score_map, geo_map, training_mask)

                self.optimizer.zero_grad()
                pred_score_map, pred_geo_map = self.model.forward(img)

                iou_loss, cls_loss = self.loss(score_map, pred_score_map, geo_map, pred_geo_map, training_mask)
                loss = iou_loss + cls_loss
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if self.verbosity >= 2:
                    self.logger.info(
                        'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} IOU Loss: {:.6f} CLS Loss: {:.6f}'.format(
                            epoch,
                            batch_idx * self.data_loader.batch_size,
                            len(self.data_loader) * self.data_loader.batch_size,
                            100.0 * batch_idx / len(self.data_loader),
                            loss.item(), iou_loss.item(), cls_loss.item()))

            except Exception:
                print(image_paths)
                raise

        # skip validation at the early stage
        if epoch < 10:
            print("skip validation...")
            metrics = {'precision': 0.0, 'recall': 0.0, 'hmean': 0.0, 'AP': 0}
        else:
            print("processing validation...")
            metrics = self.__compute_hmean()

        log = {
            'loss': total_loss / len(self.data_loader),
            'hmean': metrics['hmean'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        }

        return log

    def __compute_hmean(self):
        self.model.eval()
        temp_dir = 'temp'
        make_dir(temp_dir)
        test_img_dir = pathlib.Path(self.root_dataset) / 'test_images'
        res = main_evaluate(self.model, test_img_dir, temp_dir, self.with_gpu, False)

        return res
