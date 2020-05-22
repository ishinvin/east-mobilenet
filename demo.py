import os
import cv2
import torch
import argparse
import numpy as np

from utils.util import make_dir
from eval import load_model
from utils.util import resize_image
from utils.util import detect
from utils.util import sort_poly


def demo(model, img_path, save_path, with_gpu):
    with torch.no_grad():
        im = cv2.imread(img_path)[:, :, ::-1]
        im_resized, (ratio_h, ratio_w) = resize_image(im)
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
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                              thickness=1)

        if save_path is not None:
            print(img_path)
            cv2.imwrite(os.path.join(save_path, img_path.split('/')[-1]), im[:, :, ::-1])

        cv2.imshow('demo', im[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyWindow('demo')


def main(args: argparse.Namespace):
    with_gpu = True if torch.cuda.is_available() else False
    save_path = None
    if args.save:
        save_path = "demo"
        make_dir(save_path)
    model_path = args.model
    img_path = args.image
    model = load_model(model_path, with_gpu)
    demo(model, img_path, save_path, with_gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('-m', '--model', default=None, type=str, required=True, help='path to model')
    parser.add_argument('-i', '--image', default=None, type=str, required=True, help='input image')
    parser.add_argument('-s', '--save', default=False, type=bool, help='flag save image')
    args = parser.parse_args()
    main(args)
