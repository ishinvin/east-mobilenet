import os
import cv2
import torch
import argparse
import numpy as np
import tensorflow as tf
from scipy import stats
import openai

import logging, os
logging.disable(logging.WARNING)
logging.disable(logging.INFO)

from eval import load_model
from utils.util import (
    resize_image,
    detect,
    sort_poly,
    min_max_points,
    binary_sort,
    get_targets,
    reshape_tree,
    merge_y_axis_boxes,
    boxes_stats,
    check_dictionary,
    dictionary,
    create_prompt
)


def demo(model, img_path, save_path, with_gpu, debug_mode):
    openai.api_key = "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"
    #openai.api_key = os.getenv("OPENAI_API_KEY")
    #print(openai.api_key)
    alphaNet = tf.saved_model.load("alphaNet/1/")
    infer = alphaNet.signatures["serving_default"]
    #PDF_THRESHOLD = 0.00285
    PDF_THRESHOLD = 0.01
    target_classes = get_targets()
    show_image = True
    show_letter, show_words = debug_mode, debug_mode
    with torch.no_grad():
        im = cv2.imread(img_path)

        cv2.imshow("image", resize_image(im,928)[0])
        cv2.waitKey(0)
        cv2.destroyWindow("image")

        # Preprocess the image to pass to the word detection model

        # Inverting the color channels of the image
        # from BGR (which is the default way that cv2 returns the image) to RGB
        img = im[:, :, ::-1].copy()

        # Resizes the image to 512 in the larger side of the image 
        # and ensures that each side is a multiple of 32
        im_resized, (ratio_h, ratio_w) = resize_image(img, 512)
        im_resized = im_resized.astype(np.float32)
        im_resized = torch.from_numpy(im_resized)
        if with_gpu:
            im_resized = im_resized.cuda()

        im_resized = im_resized.unsqueeze(0)
        im_resized = im_resized.permute(0, 3, 1, 2)

        print("\nStarting prediction...")
        # Feed the image to the model
        score, geometry = model.forward(im_resized)

        # Rearrange the outputs of the model
        score = score.permute(0, 2, 3, 1)
        geometry = geometry.permute(0, 2, 3, 1)
        score = score.detach().cpu().numpy()
        geometry = geometry.detach().cpu().numpy()

        # Obtain the boxes with the highest scores 
        # and apply Non-Maximum Suppression to redundant boxes
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
                #cv2.polylines(img, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                #              thickness=2)

        #   ADDED CODE FOR THE CHARACTER RECOGNITION MODEL!!

        #print("POLYLINES")
        #print(boxes)
        
        words = ""

        # Obtain only the top left corner and bottom right corner of each box
        boxes = [min_max_points(box, im.shape[0], im.shape[1]) for box in boxes]

        # Sort each box by its initial Y coordinate first and then by its initial X coordinate
        boxes = binary_sort(boxes,1,0)[0]
        if debug_mode:
            print()
            print(boxes)
        

        for box in boxes:
            cv2.rectangle(img, box[0], box[1], (0, 255, 0), 2)
            
            # Cut the image in the location of the previously detected boxes
            letters = im[box[0][1]:box[1][1], box[0][0]:box[1][0], :].copy()

            # Preprocess the word image in order to detect the character objects in the image 
            # and obtain the bounding boxes of each character
            blur = cv2.GaussianBlur(cv2.cvtColor(letters,cv2.COLOR_BGR2GRAY),(5,5),0)
            (t,imgBin) = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)
            imgBin = cv2.bitwise_not(imgBin)
            strelDilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)) #(cv2.MORPH_ELLIPSE, (4,4))
            strelOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)) #(cv2.MORPH_ELLIPSE, (11,11))
            imgBin = cv2.dilate(imgBin,strelDilate)
            imgBin = cv2.morphologyEx(imgBin,cv2.MORPH_OPEN,strelOpen)
            
            (numLabels, labels, detection, centroids) = cv2.connectedComponentsWithStats(imgBin)

            detection = detection[1:, 0:4]
            detection = np.reshape(detection, (len(detection),2,2))

            # Calculate statistics on the sizes of all the character objects detected by cv2
            box_stats = boxes_stats(detection)

            # Invert the blacks and whites of the image 
            # to pass again to the cv2 for a second detection attempt
            img_reversed = 255 - imgBin.copy()
            detection_reversed = cv2.connectedComponentsWithStats(img_reversed)[2]
            detection_reversed = detection_reversed[1:, 0:4]
            detection_reversed = np.reshape(detection_reversed, (len(detection_reversed),2,2))
            stats_reversed = boxes_stats(detection_reversed)

            variance = box_stats[3]**2

            # This implementation is the best solution found to account for images
            # where the letters are brighter than the background
            # and for when they are darker than the background.

            # After both approaches, we consider the black and white image 
            # where the resulting detected boxes have the smaller standard deviation

            if box_stats[3] > stats_reversed[3]:
                imgBin = img_reversed
                detection = detection_reversed
                variance = stats_reversed[3]**2


            # Then we sort each bounding box of a letter 
            # from the smaller initial X position to the greater initial X position,
            # so that each letter is then appended to the final word in the correct order

            (detection, tree) = binary_sort(detection)


            # Next we implement a solution to consider only the bounding boxes of letters 
            # that are centered, relative to the word image, along the image's Y axis

            # We use a Normal Distribution where the mean is the middle of the word image 
            # and the standard deviation is a 1/10 of the image's height
            
            
            h = box[1][1] - box[0][1]

            mu, sigma = h/2, h/10
            distribution = stats.norm(mu, sigma)

            # Then we calculate the Probablity Density Function 
            # of the middle Y point of each letter's bounding box 
            # and multiply by the previously calculated variance of the boxes areas

            pdfs = [distribution.pdf(character[0][1] + (character[1][1] / 2)) * variance for character in detection]

            # We normalize the PDFs
            pdfs = pdfs / max(pdfs)

            if debug_mode:
                print("\nStatistics:")
                print("standard deviation", box_stats[3])
                print("standard deviation reversed", stats_reversed[3])
                print("pdfs: ",pdfs, "\n")
            

            # Lastly, we consider only the boxes where the pdf 
            # is greater than a certain threshold
            detection = [character for character, pdf in zip(detection,pdfs) if pdf > PDF_THRESHOLD]

            pdfs = [pdf for pdf in pdfs if pdf > PDF_THRESHOLD]

            # After that we merge boxes that are aligned along the Y axis.
            # This is to account for the cases of the 'i' and 'j' characters,
            # where cv2 detects the dot of the 'i' or the 'j' as a separate object
            detection = merge_y_axis_boxes(detection)
            
            word = ""
            
            for i in range(len(detection)):
                (x, y, w, h) = (detection[i][0][0], detection[i][0][1], detection[i][1][0], detection[i][1][1])
                letter = imgBin[y:y+h, x:x+w].copy()
                letters = cv2.rectangle(letters, (x,y),(x+w,y+h), (0,255,0),2)

                letter_resized = cv2.resize(letter,(20,20),interpolation=cv2.INTER_AREA).astype(np.float32)
                
                img_pred = np.expand_dims(letter_resized, axis=(0,3))
                outputs = infer(tf.constant(img_pred))
                y_test_pred = tf.argmax(outputs['dense_1'].numpy(), axis=1)
                index = y_test_pred.numpy()[0]
                pred = target_classes[index][str(index)].upper()
                cv2.putText(letters, pred, (int(x+(w/2)) - 9, int(y+(h/2))), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
                word += pred

                if show_letter:
                    
                    print("\n", pred)
                    print("confidence =", outputs['dense_1'].numpy()[0, y_test_pred][0])
                    print("pdf of letter location: ",pdfs[i])
                    cv2.imshow("letter", letter)
                    cv2.waitKey(0)
                    cv2.destroyWindow("letter")

            
            cv2.putText(img, word, (box[0][0], box[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            if debug_mode:
                print("\nWORD:", word)
                print("word in dictionary? ", check_dictionary(word.lower()))
                print("---------------------")

            words += word + " "
            
            if show_words:
                cv2.imshow("word", letters)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        print("\n -> PREDICTION:", words)

        prompt_string = create_prompt(words)
        #print(prompt_string)

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt_string,
            temperature=0.7,
            max_tokens=len(words),
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        result = (response["choices"][0]["text"])[1:]

        print(" -> CORRECTED:", result + "\n")

        if show_image:
            cv2.imshow(result,resize_image(img[:, :, ::-1],928)[0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        

        #print(result.replace(" ", "") + "." + (img_path.split('/')[-1]).split('.')[-1])
        #cv2.imwrite(os.path.join(save_path, 'result-{}'.format(img_path.split('/')[-1])), img[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, 'result-{}'.format(result.replace(" ", "") + "." + (img_path.split('/')[-1]).split('.')[-1])), img[:, :, ::-1])


def main(args: argparse.Namespace):
    with_gpu = True if torch.cuda.is_available() else False
    save_path = "demo"
    model_path = args.model
    img_path = args.image
    debug_mode = args.debug
    print("DEBUG MODE: ", debug_mode)
    model = load_model(model_path, with_gpu)
    demo(model, img_path, save_path, with_gpu, debug_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('-m', '--model', default=None, type=str, required=True, help='path to model')
    parser.add_argument('-i', '--image', default=None, type=str, required=True, help='input image')
    parser.add_argument('-d', '--debug', default=False, type=bool, required=False, help='debug mode')
    args = parser.parse_args()
    main(args)
