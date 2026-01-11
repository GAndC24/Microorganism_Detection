from typing import Dict, List
from PIL import Image
import cv2
import numpy as np

# Visualize the annotations on the image
def draw_boxes(image: Image, boxes: List, class_map : Dict) -> Image:
    '''在图像上绘制边界框'''
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for b in boxes:
        xmin, ymin, xmax, ymax = map(int, b["box"])
        label = class_map[b["class_id"]]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, label, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image