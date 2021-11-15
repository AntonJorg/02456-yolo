import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from model.dataloader import HELMETDataSet, HELMETDataLoader

dataloader = HELMETDataLoader("data/HELMET_DATASET", batch_size=12)

batch = next(iter(dataloader))

imgs, annotations = batch

print(imgs)
print(annotations)

fig, axs = plt.subplots(3, 4)

for (x, y), ax in np.ndenumerate(axs):
    img, annotation = imgs[4*x + y], annotations[4*x + y]
    ax.imshow(np.array(img).transpose([1, 2, 0]))

    for k, v in annotation.items():
        _, _, x, y, w, h, _ = v
        rect = plt.Rectangle((x, y), w, h, fc="none", ec="red")
        ax.add_patch(rect)

plt.show()
