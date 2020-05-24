import cv2
import torch
import pathlib
import logging
import numpy as np

from torch.utils.data import Dataset
from data.utils import crop_area
from data.utils import generate_rbox
from data.utils import check_and_validate_polys

logger = logging.getLogger(__name__)


class ICDAR(Dataset):

    def __init__(self, data_root, input_size=512):
        data_root = pathlib.Path(data_root)
        self.input_size = input_size
        self.imagesRoot = data_root / 'train_images'
        self.gtRoot = data_root / 'train_gts'
        self.images, self.bboxs, self.transcripts = self.__load_gt()

    def __load_gt(self):
        all_bboxs = []
        all_texts = []
        all_images = []
        types = ('*.jpg', '*.png', '*.JPG', '*.PNG')  # the tuple of file types
        files_grabbed = []
        for files in types:
            files_grabbed.extend(self.imagesRoot.glob(files))

        for image in files_grabbed:
            gt = self.gtRoot / image.with_name('gt_{}'.format(image.stem)).with_suffix('.txt').name
            if not gt.exists():
                continue
            all_images.append(image)
            with gt.open(mode='r') as f:
                bboxes = []
                texts = []
                for line in f:
                    text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(',')
                    x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, text[:8]))
                    bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    delim, label = '', ''
                    for i in range(9, len(text)):
                        label += delim + text[i]
                        delim = ','
                    texts.append(label.strip())
                    bboxes.append(bbox)
                bboxes = np.array(bboxes)
                all_bboxs.append(bboxes)
                all_texts.append(texts)
        return all_images, all_bboxs, all_texts

    def __getitem__(self, index):
        image_name = self.images[index]
        bboxes = self.bboxs[index]  # num_words * 8
        transcripts = self.transcripts[index]

        try:
            return self.__transform((image_name, bboxes, transcripts))
        except Exception as e:
            return self.__getitem__(torch.tensor(np.random.randint(0, len(self))))

    def __len__(self):
        return len(self.images)

    def __transform(self, gt, random_scale=np.array([0.5, 1, 2.0, 3.0]), background_ratio=3. / 8):
        """
        :param gt: iamge path (str), wordBBoxes (2 * 4 * num_words), transcripts (multiline)
        :return:
        """
        image_path, wordBBoxes, transcripts = gt
        im = cv2.imread(image_path.as_posix())
        numOfWords = len(wordBBoxes)
        text_polys = wordBBoxes  # num_words * 4 * 2
        transcripts = [word for line in transcripts for word in line.split()]
        text_tags = [True if (tag == '*' or tag == '###') else False for tag in transcripts]  # ignore '###'

        if numOfWords == len(transcripts):
            h, w, _ = im.shape
            text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

            rd_scale = np.random.choice(random_scale)
            im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
            text_polys *= rd_scale

            # random crop a area from image
            if np.random.rand() < background_ratio:
                # crop background
                im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background=True)
                if text_polys.shape[0] > 0:
                    # cannot find background
                    raise RuntimeError('cannot find background')
                # pad and resize image
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, self.input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = cv2.resize(im_padded, dsize=(self.input_size, self.input_size))
                score_map = np.zeros((self.input_size, self.input_size), dtype=np.uint8)
                geo_map_channels = 5
                #                     geo_map_channels = 5 if FLAGS.geometry == 'RBOX' else 8
                geo_map = np.zeros((self.input_size, self.input_size, geo_map_channels), dtype=np.float32)
                training_mask = np.ones((self.input_size, self.input_size), dtype=np.uint8)
            else:
                im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background=False)
                if text_polys.shape[0] == 0:
                    raise RuntimeError('cannot find background')
                h, w, _ = im.shape

                # pad the image to the training input size or the longer side of image
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, self.input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = im_padded
                # resize the image to input size
                new_h, new_w, _ = im.shape
                resize_h = self.input_size
                resize_w = self.input_size
                im = cv2.resize(im, dsize=(resize_w, resize_h))
                resize_ratio_3_x = resize_w / float(new_w)
                resize_ratio_3_y = resize_h / float(new_h)
                text_polys[:, :, 0] *= resize_ratio_3_x
                text_polys[:, :, 1] *= resize_ratio_3_y
                new_h, new_w, _ = im.shape
                score_map, geo_map, training_mask, rectangles = generate_rbox((new_h, new_w), text_polys, text_tags)

            # predict 出来的feature map 是 128 * 128， 所以 gt 需要取 /4 步长
            images = im[:, :, ::-1].astype(np.float32)  # bgr -> rgb
            score_maps = score_map[::4, ::4, np.newaxis].astype(np.float32)
            geo_maps = geo_map[::4, ::4, :].astype(np.float32)
            training_masks = training_mask[::4, ::4, np.newaxis].astype(np.float32)

            return image_path, images, score_maps, geo_maps, training_masks
        else:
            raise TypeError('Number of bboxes is inconsist with number of transcripts ')
