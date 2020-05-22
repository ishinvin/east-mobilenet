import os
import json
import torch
import logging
import pathlib
import traceback
import argparse
import subprocess

from tqdm import tqdm
from utils.util import make_dir
from utils.util import predict
from models.model import Model
from utils.eval_tool.script import default_evaluation_params, evaluate_method

logging.basicConfig(level=logging.DEBUG, format='')
logger = logging.getLogger()


def load_model(model_path, with_gpu):
    config = json.load(open('config.json'))
    logger.info("Loading checkpoint: {} ...".format(model_path))
    checkpoints = torch.load(model_path, map_location='cpu')
    if not checkpoints:
        raise RuntimeError('No checkpoint found.')
    print('Epochs: {}'.format(checkpoints['epoch']))
    state_dict = checkpoints['state_dict']
    model = Model(config)
    if with_gpu and torch.cuda.device_count() > 1:
        model.parallelize()
    model.load_state_dict(state_dict)
    if with_gpu:
        model.to(torch.device('cuda'))
    model.eval()
    return model


def evaluate(eval_dir):
    os.chdir(eval_dir)
    res = subprocess.getoutput('zip -q submit.zip *.txt')
    res = subprocess.getoutput('mv submit.zip ../')
    os.chdir('../')
    # res = subprocess.getoutput('python ./utils/eval_tool/script.py –g=./utils/eval_tool/gt.zip –s=./submit.zip')
    # print(res)

    eval_params = default_evaluation_params()
    gt_file_path = 'utils/eval_tool/gt.zip'
    subm_file_path = 'submit.zip'
    eval_data = evaluate_method(gt_file_path, subm_file_path, eval_params)
    results = eval_data['method']

    os.remove('./submit.zip')

    return results


def main_evaluate(model, input_dir, output_dir, with_gpu, with_image=False):
    types = ('*.jpg', '*.png', '*.JPG', '*.PNG')  # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(input_dir.glob(files))

    for i in tqdm(range(len(files_grabbed))):
        image_fn = files_grabbed[i]
        try:
            with torch.no_grad():
                predict(image_fn, model, with_image, output_dir, with_gpu)
        except Exception as e:
            traceback.print_exc()
            print(image_fn)

    return evaluate(output_dir)


def main(args: argparse.Namespace):
    output_dir = "outputs"
    make_dir(output_dir)
    model_path = args.model
    input_dir = args.input_dir
    with_image = args.save_img
    with_gpu = True if torch.cuda.is_available() else False
    if with_image:
        make_dir(os.path.join(output_dir, 'img'))
    model = load_model(model_path, with_gpu)

    print(main_evaluate(model, input_dir, output_dir, with_image, with_gpu))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model eval')
    parser.add_argument('-m', '--model', default=None, type=pathlib.Path, required=True, help='path to model')
    parser.add_argument('-i', '--input_dir', default=None, type=pathlib.Path, required=True, help='dir for input image')
    parser.add_argument('-s', '--save_img', default=False, type=bool, help='save images')
    args = parser.parse_args()
    main(args)
