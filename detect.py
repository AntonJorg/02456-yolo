import argparse
import os
import imageio
from tqdm import tqdm

import matplotlib.pyplot as plt
from model.dataloader import HELMETDataSet, class_dict
from model.dataloader import class_dict
from model.models import Darknet, load_weights, load_darknet_weights
from utils.utils import *


parser = argparse.ArgumentParser(description='Detect PTW riders and their helmet use on an image.\n'
                                             'Unless otherwise specified, saves results in filename_annotated.png')
parser.add_argument('filename', type=str, help='filename of the image to perform object detection on.\n'
                                               'if --dir is set, directory of files')
parser.add_argument('-s', '--suffix', type=str, help='saves the results to filename_suffix.png')
parser.add_argument('-d', '--display', action="store_true", help="whether to display the image immediately")
parser.add_argument('--nocuda', action="store_true", help="disable CUDA")
parser.add_argument('--dir', action="store_true", help="interpret filename as a directory containing the images")
parser.add_argument('--savedir', help="directory to save output to")
parser.add_argument('--gif', action="store_true", help="compile a gif of the processed images")


args = parser.parse_args()

cfg_path = './cfg/yolov3_36.cfg'
conv74_weights_path = './weights/darknet53.conv.74'
trained_weights_path = './weights/416e17.pt'


def get_darknet(img_size, cfg=cfg_path):
    return Darknet(cfg, img_size)


img_size = 416

model = get_darknet(img_size=img_size)

checkpoint = torch.load(trained_weights_path)
model.load_state_dict(checkpoint)

cuda_available = torch.cuda.is_available()
if not args.nocuda:
    if cuda_available:
        device = torch.device('cuda:0')
    else:
        print("Warning, CUDA not available.")
        device = 'cpu'
else:
    device = 'cpu'
print(f"Running on {device}")
model.to(device).eval()

if args.dir:
    filenames = os.listdir(args.filename)
    filenames = [os.path.join(args.filename, n) for n in filenames]
else:
    filenames = [args.filename]

dataset = HELMETDataSet(filenames=filenames, resize=(416, 416))
no_resize = HELMETDataSet(filenames=filenames)

opt = {
    'conf_thres': .5,  # Confidence threshold.
    'nms_thres': .45   # Non-max supression.
}

output = []
pbar = tqdm(dataset, desc="Processing images")
for i, im in enumerate(pbar):
    pbar.set_postfix({"current img": filenames[i]})
    with torch.no_grad():
        pred = model(im.unsqueeze(0).to(device))
        pred = pred[pred[:, :, 4] > opt['conf_thres']]

        if len(pred) > 0:
            detections = non_max_suppression(pred.unsqueeze(0), opt['conf_thres'], opt['nms_thres'])
            output.extend(detections)

saved_images = []

cmap = plt.cm.get_cmap("hsv", len(class_dict))

pbar = tqdm(zip(no_resize, output), desc="Saving output", total=len(dataset))
for i, (img, detection) in enumerate(pbar):
    pbar.set_postfix({"current img": filenames[i]})

    if args.savedir:
        file = os.path.split(filenames[i])[-1].split(".")[0] + "_annotated.png"
        save_path = os.path.join(args.savedir, file)
    else:
        save_path = filenames[i].split(".")[0] + "_annotated.png"

    saved_images.append(save_path)

    ### This is probably redundant in our case.
    # The amount of padding that was added
    pad_x = max(img.shape[1] - img.shape[2], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[2] - img.shape[1], 0) * (img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = 1080
    unpad_w = 1920

    # Figure
    fig = plt.figure(figsize=(19.20, 10.80), dpi=100)
    ax = fig.gca()

    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    ax.imshow(np.array(img).transpose([1, 2, 0]))
    ax.axis('off')

    # Draw bounding boxes and labels of detections
    if detection is not None:
        unique_classes = detection[:, -1].cpu().unique()

        for i in unique_classes:
            n = (detection[:, -1].cpu() == i).sum()
            # print('%g %ss' % (n, classes[int(i)]))

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
            if len(class_dict.keys()) < int(cls_pred) - 1:
                continue

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / img_size * img.shape[1]).round().item()
            box_w = ((x2 - x1) / img_size * img.shape[2]).round().item()
            y1 = (y1 / img_size * img.shape[1]).round().item()
            x1 = (x1 / img_size * img.shape[2]).round().item()

            label = list(class_dict.keys())[int(cls_pred/len(class_dict))]
            col = cmap(cls_pred)
            rect = plt.Rectangle((x1, y1), box_w, box_h, fc="none", ec=col)
            ax.add_patch(rect)
            ax.text(x1, y1, label, fontsize=10, bbox={'facecolor': col, 'pad': 2})

    plt.savefig(save_path, dpi=100)
    plt.close()

if args.gif:
    images = []

    saved_images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    pbar = tqdm(saved_images, desc="Compiling GIF")
    for filename in pbar:
        pbar.set_postfix({"current img": filename})
        images.append(imageio.imread(filename))

    if args.savedir:
        save_path = os.path.join(args.savedir, "output.gif")
    else:
        save_path = os.path.join(os.path.split(saved_images[0])[0], "output.gif")

    imageio.mimsave(save_path, images)

print("Process finished!")
