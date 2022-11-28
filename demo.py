import os
import cv2
import torch
import argparse
import numpy as np
import tensorflow as tf

import logging, os
logging.disable(logging.WARNING)
logging.disable(logging.INFO)

from eval import load_model
from utils.util import resize_image
from utils.util import detect
from utils.util import sort_poly
from utils.util import min_max_points
from utils.util import binary_sort
from utils.util import get_targets


def demo(model, img_path, save_path, with_gpu):
    alphaNet = tf.saved_model.load("alphaNet/1/")
    infer = alphaNet.signatures["serving_default"]
    target_classes = get_targets()
    show_words = True
    show_image = True
    with torch.no_grad():
        im = cv2.imread(img_path)
        img = im[:, :, ::-1].copy()
        im_resized, (ratio_h, ratio_w) = resize_image(img, 512)
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
                cv2.polylines(img, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                              thickness=2)

        #print(boxes)
        
        words = ""
        
        boxes = [min_max_points(box) for box in boxes]
        boxes = binary_sort(boxes)[0]
        #print()
        #print(boxes)

        for box in boxes:
            
            letters = im[box[0][1]:box[1][1], box[0][0]:box[1][0], :].copy()

            blur = cv2.GaussianBlur(cv2.cvtColor(letters,cv2.COLOR_BGR2GRAY),(5,5),0)
            (t,imgBin) = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)
            imgBin = cv2.bitwise_not(imgBin)
            strelDilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)) #(cv2.MORPH_ELLIPSE, (4,4))
            strelOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)) #(cv2.MORPH_ELLIPSE, (11,11))
            imgBin = cv2.dilate(imgBin,strelDilate)
            imgBin = cv2.morphologyEx(imgBin,cv2.MORPH_OPEN,strelOpen)
            imgBin = 255 - imgBin
            
           
            (numLabels, labels, detection, centroids) = cv2.connectedComponentsWithStats(imgBin)

            detection = detection[1:, 0:4]
            detection = binary_sort(np.reshape(detection, (len(detection),2,2)))[0]
            
            word = ""
            
            for i in range(len(detection)):
                (x, y, w, h) = (detection[i][0][0], detection[i][0][1], detection[i][1][0], detection[i][1][1])
                letter = imgBin[y:y+h, x:x+w].copy()
                letters = cv2.rectangle(letters, (x,y),(x+w,y+h), (0,255,0),2)
                letter = cv2.resize(letter,(20,20),interpolation=cv2.INTER_AREA).astype(np.float32)
                
                img_pred = np.expand_dims(letter, axis=(0,3))
                outputs = infer(tf.constant(img_pred))
                y_test_pred = tf.argmax(outputs['dense_1'].numpy(), axis=1)
                index = y_test_pred.numpy()[0]
                pred = target_classes[index][str(index)]
                print(pred)
                word += pred


                cv2.imshow("letter", letter)
                cv2.waitKey(0)
                cv2.destroyWindow("letter")

            cv2.putText(img, word, (box[0][0], box[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
            print("\nWORD:",word)
            words += word + " "
            
            if show_words:
                cv2.imshow("cropped", letters)
                cv2.imshow(word, imgBin)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        print("\nPREDICTION:", words)
        if show_image:
            cv2.imshow("predicted",resize_image(img[:, :, ::-1],928)[0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        

        cv2.imwrite(os.path.join(save_path, 'result-{}'.format(img_path.split('/')[-1])), img[:, :, ::-1])


def main(args: argparse.Namespace):
    with_gpu = True if torch.cuda.is_available() else False
    save_path = "demo"
    model_path = args.model
    img_path = args.image
    model = load_model(model_path, with_gpu)
    demo(model, img_path, save_path, with_gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('-m', '--model', default=None, type=str, required=True, help='path to model')
    parser.add_argument('-i', '--image', default=None, type=str, required=True, help='input image')
    args = parser.parse_args()
    main(args)
