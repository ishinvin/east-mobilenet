import cv2
import torch
import argparse
import numpy as np

from eval import load_model
from utils.util import resize_image
from utils.util import detect
from utils.util import sort_poly


def demo(model, img, with_gpu):
    im_resized, (ratio_h, ratio_w) = resize_image(img)
    im_resized = im_resized.astype(np.float32)
    im_resized = torch.from_numpy(im_resized)
    if with_gpu:
        im_resized = im_resized.cuda()

    im_resized = im_resized.unsqueeze(0)
    im_resized = im_resized.permute(0, 3, 1, 2)

    score, geometry = model.forward(im_resized)

    score = score.permute(0, 2, 3, 1)
    geometry = geometry.permute(0, 2, 3, 1)
    score = score.detach().cpu().numpy()
    geometry = geometry.detach().cpu().numpy()

    boxes = detect(score_map=score, geo_map=geometry)

    if len(boxes) > 0:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    if boxes is not None:
        for box in boxes:
            box = sort_poly(box.astype(np.int32))
            img = cv2.polylines(img, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=1)

    return img


def main(args):
    model_path = args.model
    with_gpu = True if torch.cuda.is_available() else False
    model = load_model(model_path, with_gpu)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    ret, img = cap.read()

    with torch.no_grad():
        while ret:
            ret, img = cap.read()
            if ret:
                img = demo(model, img, with_gpu)
                cv2.imshow('img', img)
                cv2.waitKey(3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model demo')
    parser.add_argument('-m', '--model', default='model_best.pth.tar', help='path to model')
    args = parser.parse_args()
    main(args)
