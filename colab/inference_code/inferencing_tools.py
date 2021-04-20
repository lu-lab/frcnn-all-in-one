import importlib
import threading
import os
import logging
from os.path import join, exists

import cv2
import h5py
import numpy as np
from utils import label_map_util
from utils import visualization_utils as vis_util


class CNN:
    ''' The CNN class loads a frozen tensorflow inference graph for object detection. The internal '_run' method runs
    the actual detection, while the outward facing 'get_worm_location' method will specifically find the center of the
     bounding-box for the highest-scoring worm object (class 1 in the provided frozen inference graph). To find all egg 
     and worm boxes, use 'get_eggs_and_worms'.'''

    def __init__(self, graph_path, labelmap_path, save_processed_images, save_dir):
        self.tf = importlib.import_module('tensorflow')
        self.box_file_lock = threading.Lock()
        self.graph_path = graph_path
        self.labelmap_path = labelmap_path
        self.save_processed_images = save_processed_images
        self.img_dir = save_dir
        if self.save_processed_images:
            self.h5_file = join(self.img_dir, 'processed', 'data.h5')
        self.cwd = os.getcwd()
        self.graph, self.sess, self.category_index = self.load_graph()
        logging.info('CNN: tensorflow model loaded')

    def load_graph(self):

        detection_graph = self.tf.Graph()
        with detection_graph.as_default():
            od_graph_def = self.tf.compat.v1.GraphDef()
            with self.tf.io.gfile.GFile(self.graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                self.tf.import_graph_def(od_graph_def, name='')

        # Load label map
        label_map = label_map_util.load_labelmap(self.labelmap_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # create a session
        sess = self.tf.compat.v1.Session(graph=detection_graph)

        return detection_graph, sess, category_index

    def _run(self, expanded_image):

        with self.graph.as_default():
            image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            scores = self.graph.get_tensor_by_name('detection_scores:0')
            classes = self.graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            (boxes, scores, classes, num_detections) = self.sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: expanded_image})

            classes = np.squeeze(classes).astype(np.int32)
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            return classes, boxes, scores

    def _prep_image(self, image):
        # change color order from BGR to RGB
        image = image[:, :, [2, 1, 0]]
        self.width = np.size(image, 1)
        self.height = np.size(image, 0)
        # print('Width is %s, Height is %s' % (self.width, self.height))
        return image

    def _screen_results(self, target_class, min_score, classes, boxes, scores):
        # screen out classes that are not 1 (worm class) and scores > .8
        idx = (classes == target_class) & (scores >= min_score)
        boxes = boxes[idx]
        scores = scores[idx]
        classes = classes[idx]
        num_results = len(boxes)

        return num_results, classes, boxes, scores

    def _get_top_result(self, num_results, classes, boxes, scores):
        # default to none
        top_box = None
        top_score = None
        top_class = None
        # if more than one worm, take the one with the highest score
        if num_results > 1:
            # scores are always ordered from highest to lowest, so...
            top_box = np.squeeze(boxes[0])
            top_score = np.squeeze(scores[0])
            top_class = np.squeeze(classes[0])
        # if one worm, get it's coordinates
        elif num_results == 1:
            top_box = np.squeeze(boxes)
            top_score = np.squeeze(scores)
            top_class = np.squeeze(classes)

        return top_class, top_box, top_score

    def _get_box_center(self, box_coords):
        ymin, xmin, ymax, xmax = box_coords
        (left, right, top, bottom) = (xmin*self.width, xmax*self.width,
                                      ymin*self.height, ymax*self.height)
        center_x = ((right - left)/2) + left
        center_y = ((bottom - top)/2) + top
        return center_x, center_y

    def get_worm_location(self, image, frame_no):
        # default to no center x or y position
        worm_center_x = None
        worm_center_y = None
        image = self._prep_image(image)
        print(frame_no)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        expanded_image = np.expand_dims(image, axis=0)
        try:
            classes, boxes, scores = self._run(expanded_image)
            target_class = 1
            min_score = 0.8
            num_worms, worm_classes, worm_boxes, worm_scores = self._screen_results(target_class, min_score,
                                                                                    classes, boxes, scores)
            worm_class, worm_box, worm_score = self._get_top_result(num_worms, worm_classes, worm_boxes, worm_scores)

            if self.save_processed_images:
                if worm_box is not None:
                    with self.box_file_lock:
                        if exists(self.h5_file):
                            with h5py.File(self.h5_file, 'a') as hf:
                                frame_boxes_name = 'worm_boxes_frame_' + str(frame_no)
                                hf.create_dataset(frame_boxes_name, data=worm_boxes)
                                frame_scores_name = 'worm_score_frame_' + str(frame_no)
                                hf.create_dataset(frame_scores_name, data=worm_scores)
                        else:
                            with h5py.File(self.h5_file, 'w') as hf:
                                frame_boxes_name = 'worm_boxes_frame_' + str(frame_no)
                                hf.create_dataset(frame_boxes_name, data=worm_boxes)
                                frame_scores_name = 'worm_score_frame_' + str(frame_no)
                                hf.create_dataset(frame_scores_name, data=worm_scores)

            if worm_box is not None:
                worm_center_x, worm_center_y = self._get_box_center(worm_box)

        except self.tf.compat.v1.errors.ResourceExhaustedError:
            # just return the center point as None, None
            pass

        return worm_box, worm_center_x, worm_center_y

    def get_eggs_and_worms(self, image, frame_no):
        worm_center_x, worm_center_y = (None, None)
        num_eggs = None
        image = self._prep_image(image)
        expanded_image = np.expand_dims(image, axis=0)
        try:
            classes, boxes, scores = self._run(expanded_image)

            target_class = 2
            min_score = 0.1
            num_eggs, egg_classes, egg_boxes, egg_scores = self._screen_results(target_class, min_score, classes, boxes,
                                                                                scores)
            target_class = 1
            min_score = 0.8
            num_worms, worm_classes, worm_boxes, worm_scores = self._screen_results(target_class, min_score,
                                                                                    classes, boxes, scores)
            if self.save_processed_images:
                with self.box_file_lock:
                    if exists(self.h5_file):
                        with h5py.File(self.h5_file, 'a') as hf:
                            if egg_boxes is not None:
                                frame_boxes_name = 'egg_boxes_frame_' + str(frame_no)
                                hf.create_dataset(frame_boxes_name, data=egg_boxes)
                                frame_scores_name = 'egg_score_frame_' + str(frame_no)
                                hf.create_dataset(frame_scores_name, data=egg_scores)
                            if worm_boxes is not None:
                                frame_boxes_name = 'worm_boxes_frame_' + str(frame_no)
                                hf.create_dataset(frame_boxes_name, data=worm_boxes)
                                frame_scores_name = 'worm_score_frame_' + str(frame_no)
                                hf.create_dataset(frame_scores_name, data=worm_scores)
                    else:
                        with h5py.File(self.h5_file, 'w') as hf:
                            if egg_boxes is not None:
                                frame_boxes_name = 'egg_boxes_frame_' + str(frame_no)
                                hf.create_dataset(frame_boxes_name, data=egg_boxes)
                                frame_scores_name = 'egg_score_frame_' + str(frame_no)
                                hf.create_dataset(frame_scores_name, data=egg_scores)
                            if worm_boxes is not None:
                                frame_boxes_name = 'worm_boxes_frame_' + str(frame_no)
                                hf.create_dataset(frame_boxes_name, data=worm_boxes)
                                frame_scores_name = 'worm_score_frame_' + str(frame_no)
                                hf.create_dataset(frame_scores_name, data=worm_scores)

        except self.tf.compat.v1.errors.ResourceExhaustedError:
            # just return the center point as None, None and the number of eggs as None
            pass

        return worm_boxes, egg_boxes

   
    
def top_worm_nn_and_unprocessed_from_h5(h5_file, unprocessed_dir, processed_dir):
    hf = h5py.File(h5_file, 'r')

    unprocessed_img_files = [f for f in os.listdir(unprocessed_dir) if os.path.isfile(os.path.join(unprocessed_dir, f))
                             and f.endswith('.jpg')]
    test_im = cv2.imread(os.path.join(unprocessed_dir,unprocessed_img_files[0]), 0)
    height, width = test_im.shape

    for f in unprocessed_img_files:
        s = f.split('img')
        r = s[1].split('.jpg')
        idx = int(r[0])
        image = cv2.imread(os.path.join(unprocessed_dir, f), 1)

        # get the boxes from hdf file
        frame_boxes_name = 'worm_boxes_frame_' + str(idx)
        score_boxes_name = 'worm_score_frame_' + str(idx)
        try:
            boxes = hf[frame_boxes_name][()]
            # screen for highest score
            if boxes.shape[0] > 1:
                scores = hf[score_boxes_name][()]
                ymin, xmin, ymax, xmax = boxes[scores == max(scores)][0]
            elif boxes.shape[0] == 1:
                ymin, xmin, ymax, xmax = boxes[0]
            new_image = cv2.rectangle(image, (int(xmin*width), int(ymin*height)), (int(xmax*width), int(ymax*height)),
                                      (25, 1, 190), 5)
            cv2.imwrite(os.path.join(processed_dir, f), new_image)
        except:
            print('Frame %s not found' % idx)
            cv2.imwrite(os.path.join(processed_dir, f), image)
            continue

            
def label_all_detections_from_h5(h5_file, source_data, save):
    ''' Label all detections of class 1 (worms) or class 2 (eggs) in a folder of .jpg images, a list of .jpg images, or a movie.
    If the sources are images, save the labeled images out as images in directory 'save', if the source is a movie, save the labeled images out as movie 'save'.'''
    
    with h5py.File(h5_file, 'r') as hf:

        # if the unprocessed source is a folder of '.jpg' images
        if os.path.isdir(source_data):
            unprocessed_img_files = [f for f in os.listdir(source_data) if os.path.isfile(os.path.join(source_data, f))
                                     and f.endswith('.jpg')]

            for f in unprocessed_img_files:
                s = f.split('img')
                r = s[1].split('.jpg')
                idx = int(r[0])
                image = cv2.imread(os.path.join(source_data, f), 1)
                xmins, ymins, xmaxs, ymaxs, ids = get_boxes_from_h5(idx, hf)
                image = label_images(xmaxs, ymaxs, xmins, ymins, ids, image)
                cv2.imwrite(os.path.join(save, f), image)
                
        elif isinstance(source_data, list):
            for f in source_data:
                r = f.split('.jpg')
                key = r[0]
                image = cv2.imread(f, 1)
                xmins, ymins, xmaxs, ymaxs, ids = get_boxes_from_h5(key, hf)
                image = label_images(xmaxs, ymaxs, xmins, ymins, ids, image)
                cv2.imwrite(os.path.join(save, f), image)

        else: # assume the source is a movie
            vid = cv2.VideoCapture(source_data)
            fps = vid.get(cv2.CAP_PROP_FPS)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            out1 = cv2.VideoWriter(save,
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                   (int(width), int(height)))
            idx = 1
            while vid.isOpened():
                ret, image = vid.read()
                if ret:
                    xmins, ymins, xmaxs, ymaxs, ids = get_boxes_from_h5(idx, hf)
                    image = label_images(xmaxs, ymaxs, xmins, ymins, ids, image)
                    out1.write(image)
                else:
                    out1.release()
                    break
                idx += 1


def label_images(xmaxs, ymaxs, xmins, ymins, ids, image):
    height, width, d = image.shape
    for i, id in enumerate(ids):
        if id == 1 and xmins[i]:  # worm
            cv2.rectangle(image, (int(xmins[i] * width), int(ymins[i] * height)),
                                  (int(xmaxs[i] * width), int(ymaxs[i] * height)),
                                  (25, 1, 190), 5)
        if id == 2 and xmins[i]:  # egg
            cv2.rectangle(image, (int(xmins[i] * width), int(ymins[i] * height)),
                                  (int(xmaxs[i] * width), int(ymaxs[i] * height)),
                                  (205, 133, 74), 5)
    return image


def get_boxes_from_h5(idx, hf):
        # get the boxes from hdf file
        worm_frame_boxes_name = 'worm_boxes_frame_' + str(idx)
        worm_score_boxes_name = 'worm_score_frame_' + str(idx)
        egg_frame_boxes_name = 'egg_boxes_frame_' + str(idx)
        egg_score_boxes_name = 'egg_score_frame_' + str(idx)
        ids = []
        ymins = []
        ymaxs = []
        xmins = []
        xmaxs = []
        try:
            worm_boxes = hf[worm_frame_boxes_name][()]
            r, c = worm_boxes.shape
            for box in range(0, r):
                ymin, xmin, ymax, xmax = worm_boxes[box]
                ymins.append(ymin)
                xmins.append(xmin)
                ymaxs.append(ymax)
                xmaxs.append(xmax)
            ids.extend(r*[1])
        except:
            print('Worms in frame %s not found' % idx)

        try:
            egg_boxes = hf[egg_frame_boxes_name][()]
            r, c = egg_boxes.shape
            for box in range(0, r):
                egg_ymin, egg_xmin, egg_ymax, egg_xmax = egg_boxes[box]
                ymins.append(egg_ymin)
                xmins.append(egg_xmin)
                ymaxs.append(egg_ymax)
                xmaxs.append(egg_xmax)
            ids.extend(r*[2])
        except:
            print('Eggs in frame %s not found' % idx)

        return xmins, ymins, xmaxs, ymaxs, ids



if __name__ == '__main__':
    h5_file = r"C:\Users\kebel\Dropbox (GaTech)\rcnn_extras\Patrick data\processed\data.h5"
    unprocessed = r"C:\Users\kebel\Dropbox (GaTech)\rcnn_extras\Patrick data\elife-38675-video3.mp4"
    processed_dir = r"C:\Users\kebel\Dropbox (GaTech)\rcnn_extras\Patrick data\processed"
    label_all_detections_from_h5(h5_file, unprocessed, processed_dir)