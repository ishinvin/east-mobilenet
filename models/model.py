import math
import torch
import torch.nn as nn
import torch.optim as optim

from base.base_model import BaseModel
from models.mobilenet_v3 import MobileNetV3


class Model:

    def __init__(self, config):
        self.detector = East(config)

    def parallelize(self):
        self.detector = torch.nn.DataParallel(self.detector)

    def to(self, device):
        self.detector = self.detector.to(device)

    def summary(self):
        self.detector.summary()

    def optimize(self, optimizer_type, params):
        optimizer = getattr(optim, optimizer_type)(
            [
                {'params': self.detector.parameters()},
            ],
            **params
        )
        return optimizer

    def train(self):
        self.detector.train()

    def eval(self):
        self.detector.eval()

    def state_dict(self):
        sd = {
            '0': self.detector.state_dict(),
        }
        return sd

    def load_state_dict(self, sd):
        self.detector.load_state_dict(sd['0'])

    @property
    def training(self):
        return self.detector.training

    def forward(self, inputs):
        """

        :param inputs:
        :return:
        """
        score_map, geo_map = self.detector(inputs)

        return score_map, geo_map


class East(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.backbone = MobileNetV3()
        self.score_map = nn.Conv2d(64, 1, kernel_size=1)
        self.geo_map = nn.Conv2d(64, 4, kernel_size=1)
        self.angle_map = nn.Conv2d(64, 1, kernel_size=1)
        self.scale = config['data_loader']['input_size']

    def forward(self, inputs):
        inputs = self.backbone(inputs)
        score = torch.sigmoid(self.score_map(inputs))
        geo_map = torch.sigmoid(self.geo_map(inputs)) * self.scale
        angle_map = (torch.sigmoid(self.angle_map(inputs)) - 0.5) * math.pi / 2
        geometry = torch.cat([geo_map, angle_map], dim=1)

        return score, geometry
