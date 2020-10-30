import numpy as np
from skimage.measure import label, regionprops
from skimage.transform import rotate
from skimage import img_as_ubyte
import cv2
import math
import glob
import imageio

path = "/Users/Mehrshad/Desktop/final_figs"

mask_img = imageio.imread(f"{path}/mask.png")
mask = np.where(mask_img == 0)
label_img = label(mask_img)
regions = regionprops(label_img)

files = glob.glob(f"{path}/map.topo.*.png")
files.sort()
for file in files:
    image = imageio.imread(file)
    image[mask[0], mask[1], :] = 0
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        rotated = img_as_ubyte(rotate(image[min_row: max_row, min_col: max_col, :],
                                      angle=90-math.degrees(region.orientation), order=0))
        _, thresh = cv2.threshold(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[-1])
        crop = rotated[y:y + h, x:x + w, :]
        cv2.imwrite(file.replace(".png", f".reg{region.label}.png").replace("map.topo", "arrow."), crop)
