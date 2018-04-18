import os
import cv2
import sys
import argparse
import numpy as np
import coco
import utils
import model as modellib
from classes import get_class_names, InferenceConfig
from ast import literal_eval as make_tuple
import imageio
import visualize

def create_mask_color(image, color):

    color_mask = np.full(shape=(image.shape[0], image.shape[1], 3),
                         fill_value=color)
    return color_mask

def dilate_mask(mask_selected):

    kernel = np.ones((7, 7), np.uint8)
    dilated_mask = cv2.dilate(mask_selected, kernel, iterations=2)
    return dilated_mask

def person_blocker(args):

    # Required to load model, but otherwise unused
    ROOT_DIR = os.getcwd()
    COCO_MODEL_PATH = args.model or os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")  # Required to load model

    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Load model and config
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    image = imageio.imread(args.image)

    # Create masks for all objects
    results = model.detect([image], verbose=0)
    r = results[0]

    # Filter masks to only the selected objects
    objects = np.array(args.objects)

    # Object IDs:
    if np.all(np.chararray.isnumeric(objects)):
        object_indices = objects.astype(int)
    # Types of objects:
    else:
        selected_class_ids = np.flatnonzero(np.in1d(get_class_names(),
                                                    objects))
        object_indices = np.flatnonzero(
            np.in1d(r['class_ids'], selected_class_ids))

    mask_selected = np.sum(r['masks'][:, :, object_indices], axis=2)
    mask_selected = mask_selected.astype(np.uint8)

    # Dilate Mask
    mask_selected = dilate_mask(mask_selected)

    # Replace object masks with noise
    # image_masked = image.copy()
    # mask = create_mask_color(image, [255, 255, 255])
    # image_masked[mask_selected > 0] = mask[mask_selected > 0]

    # imageio.imwrite(args.output, image_masked)
    mask_selected[mask_selected > 0] = 255
    mask_selected = 255 - mask_selected
    imageio.imwrite(args.output, mask_selected)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Person Blocker - Automatically "block" people '
                    'in images using a neural network.')
    parser.add_argument('-i', '--image',  help='Image file name.',
                        required=False)
    parser.add_argument('--output', help='Output Image file name.',
                        required=False)
    parser.add_argument(
        '-m', '--model',  help='path to COCO model', default=None)
    parser.add_argument('-o',
                        '--objects', nargs='+',
                        help='object(s)/object ID(s) to block. ' +
                        'Use the -names flag to print a list of ' +
                        'valid objects',
                        default='person')
    parser.add_argument('-n',
                        '--names', dest='names',
                        action='store_true',
                        help='prints class names and exits.')
    parser.set_defaults(names=False)
    args = parser.parse_args()

    if args.names:
        print(get_class_names())
        sys.exit()

    person_blocker(args)
