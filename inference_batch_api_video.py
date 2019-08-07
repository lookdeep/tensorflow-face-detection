#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import requests

import json
import sys
import time
import numpy as np
import cv2

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils_color as vis_util



# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

cap = cv2.VideoCapture("./media/test.mp4")
X_RES = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Y_RES = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(
    "./media/test_out.avi",
    0,
    25.0,
    (X_RES, Y_RES),
)

def image_preprocess(img, expand=False):
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if expand:
        image_np = np.expand_dims(image_np, axis=0)
    return image_np


BATCH_SIZE = 8
MAX_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# batching by BATCH_SIZE
for frame in range(0, MAX_FRAMES, BATCH_SIZE):
    frames = []
    for i in range(BATCH_SIZE):
        ret, image = cap.read()
        if ret == 0:
            break
        frames.append(image)

    # Actual detection.

    request_frames = [image_preprocess(im).tolist() for im in frames]
    #request_frames = np.array(frames).reshape((BATCH_SIZE, Y_RES, X_RES, 3)).tolist()
    start_time = time.time()
    response = requests.post(
        'http://{}:18501/v1/models/face:predict'.format('127.0.0.1'),
        data=json.dumps({
            'instances': request_frames,
        }),
    )
    response.raise_for_status()
    preds = response.json()['predictions']

    #(boxes, scores, classes, num_detections) = sess.run(
    #    [boxes, scores, classes, num_detections],
    #    feed_dict={image_tensor: image_np_expanded})
    elapsed_time = time.time() - start_time
    print('batched inference time cost: {}'.format(elapsed_time))
    #print(boxes.shape, boxes)
    #print(scores.shape,scores)
    #print(classes.shape,classes)
    #print(num_detections)
    # Visualization of the results of a detection.
    for i, pred in enumerate(preds):
        vis_util.visualize_boxes_and_labels_on_image_array(
    #          image_np,
            frames[i],
            np.squeeze(pred['boxes']),
            np.squeeze(pred['classes']).astype(np.int32),
            np.squeeze(pred['scores']),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4)
        out.write(frames[i])

# now the leftovers for max_frames % BATCH_SIZE
for frame in range(MAX_FRAMES - (MAX_FRAMES % BATCH_SIZE), MAX_FRAMES):
    ret, image = cap.read()
    if ret == 0:
        break

    if out is None:
        [h, w] = image.shape[:2]
        out = cv2.VideoWriter("./media/test_out.avi", 0, 25.0, (w, h))


    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    start_time = time.time()
    response = requests.post(
        'http://{}:18501/v1/models/face:predict'.format('127.0.0.1'),
        data=json.dumps({
            'instances': image_np_expanded.tolist(),
        }),
    )
    response.raise_for_status()
    pred = response.json()['predictions'][0]

    elapsed_time = time.time() - start_time
    print('inference time cost: {}'.format(elapsed_time))
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(pred['boxes']),
        np.squeeze(pred['classes']).astype(np.int32),
        np.squeeze(pred['scores']),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4)
    out.write(image)



cap.release()
out.release()
