import importlib
import threading
import os
import logging
from os.path import join, exists

import cv2
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


class CNN:
    ''' The CNN class loads a frozen tensorflow inference graph for object detection. The internal '_run' method runs
    the actual detection, while the outward facing 'get_worm_location' method will specifically find the center of the
     bounding-box for the highest-scoring worm object (class 1 in the provided frozen inference graph). To find all egg 
     and worm boxes, use 'get_eggs_and_worms'.'''

    def __init__(self, graph_path, labelmap_path, save_processed_images, h5_file):
        self.box_file_lock = threading.Lock()
        self.graph_path = graph_path
        self.labelmap_path = labelmap_path
        self.save_processed_images = save_processed_images
        self.h5_file = h5_file
        self.cwd = os.getcwd()

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
        (left, right, top, bottom) = (xmin * self.width, xmax * self.width,
                                      ymin * self.height, ymax * self.height)
        center_x = ((right - left) / 2) + left
        center_y = ((bottom - top) / 2) + top
        return center_x, center_y

    def _prep_image(self, image):
        # change color order from BGR to RGB
        image = image[:, :, [2, 1, 0]]
        self.width = np.size(image, 1)
        self.height = np.size(image, 0)
        # print('Width is %s, Height is %s' % (self.width, self.height))
        return image

    def _screen_results(self, target_class, min_score, classes, boxes, scores):
        # screen out classes that are not target class and scores less than the minimum score
        idx = (classes == target_class) & (scores >= min_score)
        boxes = boxes[idx]
        scores = scores[idx]
        classes = classes[idx]
        num_results = len(boxes)

        return num_results, classes, boxes, scores

    def make_hdf(self):
        with h5py.File(self.h5_file, 'w') as hf:
            for k, v, in self.label_map_dict.items():
                hf.create_dataset('label_'+ str(v), data=k)

    def get_detections(self, image, frame_no, target_classes, target_min_scores):
        classes = {}
        boxes = {}
        scores = {}
        num_detections = {}
        image = self._prep_image(image)
        try:
            unfiltered_classes, unfiltered_boxes, unfiltered_scores = self._run(image)

            for target_class, target_min_score in zip(target_classes, target_min_scores):

                num_detections[target_class], classes[target_class], boxes[target_class], scores[target_class] = \
                    self._screen_results(target_class,
                                         target_min_score,
                                         unfiltered_classes,
                                         unfiltered_boxes,
                                         unfiltered_scores)

            if self.save_processed_images:
                with self.box_file_lock:
                    if exists(self.h5_file):
                        with h5py.File(self.h5_file, 'a') as hf:
                            for target_class in target_classes:
                                if num_detections[target_class] > 0:
                                    name = self.category_index[target_class+1]['name']
                                    frame_boxes_name = name + '_boxes_frame_' + str(frame_no)
                                    hf.create_dataset(frame_boxes_name, data=boxes[target_class])
                                    frame_scores_name = name + '_score_frame_' + str(frame_no)
                                    hf.create_dataset(frame_scores_name, data=scores[target_class])

        except tf.compat.v1.errors.ResourceExhaustedError:
            # just return the center point as None, None and the number of eggs as None
            pass

        return boxes, unfiltered_classes, unfiltered_boxes, unfiltered_scores


class CNN_tf1(CNN):
    ''' The CNN class loads a saved TF1 model for object detection. The internal '_run' method runs
    the actual detection, while the outward facing 'get_worm_location' method will specifically find the center of the
     bounding-box for the highest-scoring worm object (class 1 in the saved model). '''

    def __init__(self, graph_path, labelmap_path, save_processed_images, h5_file):
        super().__init__(graph_path, labelmap_path, save_processed_images, h5_file)
        self.detect_fn, self.category_index, self.label_map_dict = self.load_graph()
        self.make_hdf()
        logging.info('CNN: tensorflow 1 model loaded')

    def load_graph(self):
        model = tf.saved_model.load(self.graph_path)
        detect_fn = model.signatures['serving_default']

        # Load label map
        label_map = label_map_util.load_labelmap(self.labelmap_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        label_map_dict = label_map_util.get_label_map_dict(self.labelmap_path)

        return detect_fn, category_index, label_map_dict

    def _run(self, image):
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = self.detect_fn(input_tensor)
        detections['detection_classes'] = detections['detection_classes:0'][0]
        classes = tf.cast(detections['detection_classes'], tf.int64)
        boxes = detections['detection_boxes:0'][0].numpy()
        scores = detections['detection_scores:0'][0].numpy()

        return classes, boxes, scores

    def _label_image(self, image, box, score, class_idx=1):
        ymin, xmin, ymax, xmax = box
        class_label = self.category_index[class_idx]['name']
        display_str = '{}: {}%'.format(class_label, int(100 * score))
        vis_util.draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color='aquamarine',
            thickness=8,
            display_str_list=[display_str],
            use_normalized_coordinates=True)
        return image


class CNN_tf2(CNN):

    def __init__(self, graph_path, labelmap_path, save_processed_images, h5_file, config_path):
        super().__init__(graph_path, labelmap_path, save_processed_images, h5_file)
        self.config_path = config_path
        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        self.detection_model, self.category_index, self.label_map_dict = self.load_graph()
        self.make_hdf()
        logging.info('CNN: tensorflow 2 model loaded')

    def load_graph(self):
        configs = config_util.get_configs_from_pipeline_file(self.config_path)
        model_config = configs['model']
        detection_model = model_builder.build(model_config=model_config, is_training=False)
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(self.graph_path, 'ckpt-0')).expect_partial()
        category_index = label_map_util.create_category_index_from_labelmap(self.labelmap_path,
                                                                            use_display_name=True)
        label_map_dict = label_map_util.get_label_map_dict(self.labelmap_path)

        return detection_model, category_index, label_map_dict

    @tf.function
    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)

        return detections

    def _run(self, image):
        expanded_image = np.expand_dims(image, axis=0)
        input_tensor = tf.convert_to_tensor(expanded_image, dtype=tf.float32)
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        return detections['detection_classes'].astype(np.int64), detections['detection_boxes'], detections['detection_scores']


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

            
def label_all_detections_from_h5(h5_file, source_data, save, target_classes):
    ''' Label all detections for classes in list target_classes in a folder of .jpg images, a list of .jpg images, or a movie.
    If the sources are images, save the labeled images out as images in directory 'save', if the source is a movie, save the labeled images out as movie 'save'.'''

    ext_list = ['.jpg', '.png', '.tiff']
    with h5py.File(h5_file, 'r') as hf:

        # if the unprocessed source is a folder of '.jpg' images
        if isinstance(source_data, list):
            for f in source_data:
                key = os.path.basename(f)
                image = cv2.imread(f, 1)
                xmins, ymins, xmaxs, ymaxs, ids, _ = get_boxes_from_h5(key, hf, target_classes)
                image = label_images(xmaxs, ymaxs, xmins, ymins, ids, image)
                cv2.imwrite(os.path.join(save, key), image)
        elif os.path.isdir(source_data):
            unprocessed_img_files = [f for f in os.listdir(source_data) if os.path.isfile(os.path.join(source_data, f))
                                     and f.endswith(tuple(ext_list))]
            for f in unprocessed_img_files:
                key = os.path.basename(f)
                image = cv2.imread(os.path.join(source_data, f), 1)
                xmins, ymins, xmaxs, ymaxs, ids, _ = get_boxes_from_h5(key, hf, target_classes)
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
                    xmins, ymins, xmaxs, ymaxs, ids, _ = get_boxes_from_h5(idx, hf, target_classes)
                    image = label_images(xmaxs, ymaxs, xmins, ymins, ids, image)
                    out1.write(image)
                else:
                    out1.release()
                    break
                idx += 1


def label_images(xmaxs, ymaxs, xmins, ymins, ids, image):
    height, width, d = image.shape
    cmap = plt.get_cmap('tab10')

    for i, id in enumerate(ids):
        rgba = cmap(((id-1) % 10)/10)
        bgr = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
        cv2.rectangle(image, 
                      (int(xmins[i] * width), int(ymins[i] * height)),
                      (int(xmaxs[i] * width), int(ymaxs[i] * height)),
                      bgr,
                      5)

    return image


def get_boxes_from_h5(idx, hf, target_classes):
    ids = []
    ymins = []
    ymaxs = []
    xmins = []
    xmaxs = []
    scores = []
    for t_class in target_classes:
        # get the name of the class
        t_name = hf['label_' + str(t_class + 1)][()].decode("utf-8")
        # get the boxes from hdf file
        frame_boxes_name = t_name + '_boxes_frame_' + str(idx)
        score_boxes_name = t_name + '_score_frame_' + str(idx)

        try:
            i_boxes = hf[frame_boxes_name][()]
            i_scores = hf[score_boxes_name][()]
            r, c = i_boxes.shape
            for box in range(0, r):
                ymin, xmin, ymax, xmax = i_boxes[box]
                ymins.append(ymin)
                xmins.append(xmin)
                ymaxs.append(ymax)
                xmaxs.append(xmax)
                scores.append(i_scores[box])
            ids.extend(r*[t_class+1])
        except:
            print('%s in frame %s not found' % (t_name, idx))

        return xmins, ymins, xmaxs, ymaxs, ids, scores



if __name__ == '__main__':
    h5_file = r"C:\Users\kebel\Dropbox (GaTech)\rcnn_extras\Patrick data\processed\data.h5"
    unprocessed = r"C:\Users\kebel\Dropbox (GaTech)\rcnn_extras\Patrick data\elife-38675-video3.mp4"
    processed_dir = r"C:\Users\kebel\Dropbox (GaTech)\rcnn_extras\Patrick data\processed"
    label_all_detections_from_h5(h5_file, unprocessed, processed_dir)
