import matplotlib.pyplot as plt
import numpy as np

from model.dataloader import HELMETDataSet, HELMETDataLoader, class_dict

report_examples = ["Mandalay_1_118/93.jpg",
                   "Mandalay_2_57/40.jpg",
                   "NyaungU_rural_32/23.jpg",
                   "Pathein_rural_2/86.jpg"]

dataset = HELMETDataSet(root_dir="data/HELMET_DATASET", split="test")

idx = []

for i, p in enumerate(dataset.images_paths):
    for ex in report_examples:
        if ex in p:
            print(i, p)
            idx.append(i)

cmap = plt.cm.get_cmap("hsv", len(class_dict))

for i in idx:
    img, targets, annotation = dataset[i]

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

    for _, a in annotation.items():
        x, y, w, h = a["dim"]

        label = a["class"]
        col = cmap(class_dict[label])
        rect = plt.Rectangle((x, y), w, h, fc="none", ec=col, linewidth=8)
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=15, bbox={'facecolor': col, 'pad': 2})

    plt.savefig(f"output/report_ex_{i}.jpg", dpi=100)
    plt.close()
