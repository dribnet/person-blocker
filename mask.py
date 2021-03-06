import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
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

# Creates a color layer and adds Gaussian noise.
# For each pixel, the same noise value is added to each channel
# to mitigate hue shfting.


def create_noisy_color(image, color):
    color_mask = np.full(shape=(image.shape[0], image.shape[1], 3),
                         fill_value=color)

    noise = np.zeros((image.shape[0], image.shape[1]))
    # noise = np.random.normal(0, 25, (image.shape[0], image.shape[1]))
    noise = np.repeat(np.expand_dims(noise, axis=2), repeats=3, axis=2)
    mask_noise = np.clip(color_mask + noise, 0., 255.)
    return mask_noise


# Helper function to allow both RGB triplet + hex CL input

def string_to_rgb_triplet(triplet):

    if '#' in triplet:
        # http://stackoverflow.com/a/4296727
        triplet = triplet.lstrip('#')
        _NUMERALS = '0123456789abcdefABCDEF'
        _HEXDEC = {v: int(v, 16)
                   for v in (x + y for x in _NUMERALS for y in _NUMERALS)}
        return (_HEXDEC[triplet[0:2]], _HEXDEC[triplet[2:4]],
                _HEXDEC[triplet[4:6]])

    else:
        # https://stackoverflow.com/a/9763133
        triplet = make_tuple(triplet)
        return triplet


def person_blocker(args):

    # Required to load model, but otherwise unused
    ROOT_DIR = os.getcwd()
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    COCO_MODEL_PATH = args.model or os.path.join(SCRIPT_DIR, "mask_rcnn_coco.h5")

    MODEL_DIR = os.path.join(SCRIPT_DIR, "logs")  # Required to load model

    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Load model and config
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    image = imageio.imread(args.infile)

    # Create masks for all objects
    results = model.detect([image], verbose=0)
    r = results[0]

    dirname = os.path.dirname(args.infile)
    basename = os.path.basename(args.infile)
    barename = os.path.splitext(basename)[0]

    if args.objects is None:
        # now paste it all together
        outfile = os.path.join(dirname, "labels_{}.png".format(barename))

        position_ids = ['[{}]'.format(x)
                        for x in range(r['class_ids'].shape[0])]
        print("Saving: {}".format(outfile))
        visualize.display_instances(image, r['rois'],
                                    r['masks'], r['class_ids'],
                                    get_class_names(), position_ids, outfile=outfile)
        sys.exit()

    # Object IDs:
    indices_list = []
    objects_list = list(args.objects.split(","))
    for objects_entry in objects_list:
        if np.chararray.isnumeric(objects_entry):
            indices_list += [objects_entry]

        else:
            selected_class_ids = np.flatnonzero(np.in1d(get_class_names(),
                                                        [objects_entry]))
            indices_list += np.flatnonzero(
                np.in1d(r['class_ids'], selected_class_ids)).tolist()
    object_indices = np.asarray(indices_list).astype(int)

    mask_selected = np.sum(r['masks'][:, :, object_indices], axis=2)

    # Replace object masks with noise
    mask_colors = list(map(string_to_rgb_triplet,args.colors.split(",")))
    if args.bgcolor.lower() != "none":
        bg_color = string_to_rgb_triplet(args.bgcolor)
        image_masked = np.full(shape=(image.shape[0], image.shape[1], 3),
                         fill_value=bg_color).astype(np.uint8)
    else:
        image_masked = image.copy()

    noisy_color = create_noisy_color(image, mask_colors[0])
    image_masked[mask_selected > 0] = noisy_color[mask_selected > 0]

    outfile = os.path.join(dirname, "mask_{}.png".format(barename))
    print("Saving: {}".format(outfile))
    imageio.imwrite(outfile, image_masked)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Person Blocker - Automatically "block" people '
                    'in images using a neural network.')

    parser.add_argument(
        '-m', '--model',  help='path to COCO model', default=None)

    parser.add_argument('-c',
                        '--colors', default='#ffffff',
                        help='colors of the masks')

    parser.add_argument('-b',
                        '--bgcolor', default='#000000',
                        help='background color')

    parser.add_argument('infile', nargs='?', type=str,
                         default=None)
    parser.add_argument('objects', nargs='?', type=str,
                         default=None)
    parser.add_argument('max', nargs='?', type=int,
                         default=None, help="maximum number of objects")
    # parser.add_argument('-i', '--image',  help='Image file name.',
    #                     required=False)
    # parser.add_argument('-o',
    #                     '--objects', nargs='+',
    #                     help='object(s)/object ID(s) to block. ' +
    #                     'Use the -names flag to print a list of ' +
    #                     'valid objects',
    #                     default='person')
    # parser.add_argument('-l',
    #                     '--labeled', dest='labeled',
    #                     action='store_true',
    #                     help='generate labeled image instead')
    # parser.add_argument('-n',
    #                     '--names', dest='names',
    #                     action='store_true',
    #                     help='prints class names and exits.')
    # parser.set_defaults(labeled=False, names=False)
    args = parser.parse_args()

    if args.infile is None:
        print("All possible classes:")
        print(get_class_names())
        sys.exit()

    print("infile: {}".format(args.infile))
    print("objects: {}".format(args.objects))
    # print("max: {}".format(args.max))

    person_blocker(args)
