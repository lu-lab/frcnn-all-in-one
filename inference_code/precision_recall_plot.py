import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = r"C:\Users\kebel\Dropbox (GaTech)\mipi_exp\mipi_documentation\neural nets\object_detectors\tensorflow model 1\training\label_map.pbtxt"

# Number of classes to detect
NUM_CLASSES = 2

OUTPLOT = r"C:\Users\kebel\Dropbox (GaTech)\mipi_exp\mipi_documentation\neural nets\object_detectors\tensorflow model 1\precision_recall_tf_faster_rcnn_v1_worms_test_detections.pdf"


# function to calculate IoU

def IoU(bounding_box_p, bounding_box_gt):
    bbox_gt_top, bbox_gt_left, bbox_gt_bottom, bbox_gt_right = bounding_box_gt
    bbox_p_top, bbox_p_left, bbox_p_bottom, bbox_p_right = bounding_box_p
    x_overlap = max(0, min(bbox_gt_right, bbox_p_right) - max(bbox_gt_left, bbox_p_left))
    y_overlap = max(0, min(bbox_gt_bottom, bbox_p_bottom) - max(bbox_gt_top, bbox_p_top))
    intersection = x_overlap * y_overlap
    union = (bbox_gt_right - bbox_gt_left) * (bbox_gt_bottom - bbox_gt_top) + \
            (bbox_p_right - bbox_p_left) * (bbox_p_bottom - bbox_p_top) - intersection

    return intersection / union



# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(
#     label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)


# load detections
detection_df = pd.read_csv(r"C:\Users\kebel\Dropbox (GaTech)\mipi_exp\mipi_documentation\neural nets\object_detectors\tensorflow model 1\all_test_detections.csv")
# load ground truths
gt_df = pd.read_csv(r"C:\Users\kebel\Dropbox (GaTech)\mipi_exp\mipi_documentation\neural nets\object_detectors\tensorflow model 1\annotations\test_labels.csv")
# set thresholds
iou_threshold = 0.5
score_threshold = 0.5

# order detections by their score
detection_df = detection_df.sort_values('score', ascending=False)
precision = []
recall = []
cum_true_positives = []
false_positive_labels = []
true_positive_labels = []

unique_filenames = gt_df.filename.unique()
num_gt = 0
for fn in unique_filenames:
    if any(detection_df.image_name.str.contains(fn)):
        im_gt = gt_df.loc[lambda df: df.filename == fn]
        worm_gt = im_gt.loc[lambda df: df['class'] == 'worm']
        num_gt += worm_gt.shape[0]

for detection in detection_df.itertuples(index=False):
    # only look at worms
    if int(detection.id) == 1:
        if detection.score > score_threshold:
            # filter gt detections by same class in same image
            im_gt = gt_df.loc[lambda df: df.filename == detection.image_name]
            if im_gt.empty is True:
                # im_gt should only be empty if the image the detection was in isn't in the ground truth set
                continue
            worm_gt = im_gt.loc[lambda df: df['class'] == 'worm']
            #egg_gt = im_gt.loc[lambda df: df['class'] == 'egg']
            # calculate ioU between each filtered ground truth and detection
            iou = []
            for i_worm, worm in enumerate(worm_gt.itertuples()):
                iou.append(IoU((detection.ymin, detection.xmin, detection.ymax, detection.xmax),
                               (worm.ymin / worm.height, worm.xmin / worm.width,
                                worm.ymax / worm.height, worm.xmax / worm.width)))
            # if no ground-truth can be chosen or IoU < threshold (e.g., 0.5)
            if worm_gt.empty is True or max(iou) < iou_threshold:
                # the detection is a false positive
                false_positive_labels.append(1)
                true_positive_labels.append(0)
            else:
                # the detection is a true positive
                max_iou = max(iou)
                false_positive_labels.append(0)
                true_positive_labels.append(1)

cum_true_positives = np.cumsum(true_positive_labels)
cum_false_positives = np.cumsum(false_positive_labels)
precision = cum_true_positives.astype(float) / (cum_true_positives + cum_false_positives)
recall = cum_true_positives.astype(float) / num_gt
pr_data = {'precision': precision, 'recall': recall}
pr_df = pd.DataFrame(data=pr_data)
print('Number of ground truth observations: %s' % num_gt)

sns.set(style="white")
ax = sns.scatterplot(x='recall', y='precision', data=pr_data, ci=None)
ax.set_xlabel('recall')
ax.set_ylabel('precision')
# find all unique recall values, for each value use the max precision to interpolate
unique_recalls = pr_df.recall.unique()
shift_unique_recalls = [0]
shift_unique_recalls.extend(unique_recalls[:-1])
p_interp = []
integral = 0
# calculate area under the curve to get average precision
for recall_level, last_recall_level in zip(unique_recalls, shift_unique_recalls):
    recall_level_df = pr_df.loc[lambda df: df.recall == recall_level]
    p_interp.append(recall_level_df.precision.max())
    integral += (recall_level - last_recall_level) * recall_level_df.precision.max()

ax = sns.lineplot(x=unique_recalls, y=p_interp, ci=None)
plt.ylim(0, 1.15)
plt.xlim(0, 1)
plt.savefig(OUTPLOT, transparent=True)
# integral
print('precision for worms at a score threshold of %s and an iou threshold of %s: %s' % (score_threshold, iou_threshold, integral))
