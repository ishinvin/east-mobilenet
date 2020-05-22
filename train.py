import os
import json
import logging
import argparse

from models.loss import Loss
from models.model import Model
from logger.logger import Logger
from data.loader import ICDARDataLoader
from trainer.trainer import Trainer

logging.basicConfig(level=logging.DEBUG, format='')


def main(config, resume):
    train_logger = Logger()

    # load data
    train_dataloader = ICDARDataLoader(config).train()

    # initial model
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config['gpus']])
    model = Model(config)
    model.summary()

    loss = Loss()
    trainer = Trainer(model, loss, resume, config, train_dataloader, train_logger)
    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='EAST')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args = parser.parse_args()
    config = json.load(open('config.json'))
    if args.resume:
        logger.warning('Warning: --config overridden by --resume')

    main(config, args.resume)
