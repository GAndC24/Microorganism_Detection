import xml.etree.ElementTree as ET
from typing import Dict, Tuple, List, TypedDict
from dataclasses import dataclass
from PIL import Image
import cv2
import numpy as np

Box = Tuple[float, float, float, float]   # (x1, y1, x2, y2)

class LabeledBox(TypedDict):
    box: Box
    class_id: int

# read the VOC XML annotation file
def parse_voc_xml(xml_path: str, class_map : Dict) -> Tuple[Dict, List[LabeledBox]]:
    '''读取 VOC XML 标注文件并返回 image_info 和 boxes'''
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 读取 image_info
    size = root.find("size")
    image_info = {
        "width": int(size.find("width").text),
        "height": int(size.find("height").text),
        "depth": int(size.find("depth").text),
    }

    # 读取 boxes
    boxes = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")

        class_id = class_map[label]

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        gt : LabeledBox = {
            "box" : (xmin, ymin, xmax, ymax),
            "class_id" : class_id
        }
        boxes.append(gt)

    return image_info, boxes

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