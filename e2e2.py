# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import cv2
import numpy as np
import json
import yaml
import onnxruntime as ort
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from tools.infer.utility import get_rotate_crop_image

# Add PaddleOCR to system path
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

# Setup logging
logger = get_logger()
import numpy as np
import cv2
import numpy as np
import cv2

def calculate_iou(box1, box2):
    """Tính IoU giữa hai box (dạng tọa độ 4 điểm)."""
    # Chuyển box thành dạng [x_min, y_min, x_max, y_max]
    x1_min, y1_min = np.min(box1, axis=0)
    x1_max, y1_max = np.max(box1, axis=0)
    x2_min, y2_min = np.min(box2, axis=0)
    x2_max, y2_max = np.max(box2, axis=0)

    # Tính diện tích giao nhau
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Tính diện tích hợp
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

def is_box_inside(box_small, box_large, threshold=0.9):
    """Kiểm tra nếu box_small nằm trong box_large (dựa trên diện tích giao nhau)."""
    iou = calculate_iou(box_small, box_large)
    # Tính diện tích box_small
    area_small = np.abs((np.max(box_small, axis=0)[0] - np.min(box_small, axis=0)[0]) *
                        (np.max(box_small, axis=0)[1] - np.min(box_small, axis=0)[1]))
    # Tính diện tích giao nhau
    inter_area = iou * area_small / (1 - iou + 1e-10)  # Tránh chia cho 0
    # Nếu diện tích giao nhau chiếm hơn threshold của box_small, coi là box_small nằm trong box_large
    return inter_area / area_small > threshold if area_small > 0 else False

class OCRProcessor:
    def __init__(self, det_config_path, det_onnx_path, rec_config_path, rec_onnx_path):
        """Khởi tạo mô hình phát hiện và nhận diện văn bản một lần với ONNX."""
        # Load detection configuration
        with open(det_config_path, 'r') as f:
            self.det_config = yaml.safe_load(f)

        # Load detection ONNX model
        self.det_session = ort.InferenceSession(det_onnx_path)
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.det_output_name = self.det_session.get_outputs()[0].name

        # Build detection post-process
        self.det_post_process = build_post_process(self.det_config["PostProcess"])
        
        # Setup detection transforms
        det_transforms = []
        for op in self.det_config["Eval"]["dataset"]["transforms"]:
            op_name = list(op)[0]
            if "Label" in op_name:
                continue
            elif op_name == "KeepKeys":
                op[op_name]["keep_keys"] = ["image", "shape"]
            det_transforms.append(op)
        self.det_ops = create_operators(det_transforms, self.det_config["Global"])

        # Load recognition configuration
        with open(rec_config_path, 'r') as f:
            self.rec_config = yaml.safe_load(f)
        self.rec_global_config = self.rec_config["Global"]

        # Load recognition ONNX model
        self.rec_session = ort.InferenceSession(rec_onnx_path)
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        self.rec_output_name = self.rec_session.get_outputs()[0].name

        # Build recognition post-process
        self.rec_post_process = build_post_process(self.rec_config["PostProcess"], self.rec_global_config)

        # Setup recognition transforms
        rec_transforms = []
        for op in self.rec_config["Eval"]["dataset"]["transforms"]:
            op_name = list(op)[0]
            if "Label" in op_name:
                continue
            elif op_name in ["RecResizeImg"]:
                op[op_name]["infer_mode"] = True
            elif op_name == "KeepKeys":
                op[op_name]["keep_keys"] = ["image"]
            rec_transforms.append(op)
        self.rec_global_config["infer_mode"] = True
        self.rec_ops = create_operators(rec_transforms, self.rec_global_config)

        
    def detect_text(self, image):
        """Phát hiện văn bản trên ảnh, sử dụng mô hình ONNX đã tải sẵn."""
        if image is None:
            print("Ảnh đầu vào không hợp lệ")
            return None, None

        # Convert image to bytes for processing
        _, img_data = cv2.imencode('.png', image)
        data = {"image": img_data.tobytes()}
        batch = transform(data, self.det_ops)

        # Prepare for ONNX model
        images = np.expand_dims(batch[0], axis=0).astype(np.float32)
        shape_list = np.expand_dims(batch[1], axis=0)
        
        # Run detection with ONNX
        preds = self.det_session.run([self.det_output_name], {self.det_input_name: images})[0]
        post_result = self.det_post_process({"maps": preds}, shape_list)

        # Process detection results
        if isinstance(post_result, dict):
            boxes = post_result[list(post_result.keys())[0]][0]["points"]
        else:
            boxes = post_result[0]["points"]

        if not len(boxes):
            print("Không phát hiện box nào")
            return None

        # Nếu chỉ có 1 box, trả về ngay
        if len(boxes) == 1:
            box = boxes[0]
            cropped_img = crop_and_straighten_image(image, box)
            return cropped_img

        # Xử lý nhiều box: lọc box trùng và box nhỏ trong box lớn
        filtered_boxes = []
        iou_threshold = 0.3  # Ngưỡng IoU để coi là trùng
        inside_threshold = 0.6  # Ngưỡng để coi box nhỏ nằm trong box lớn

        # Tính diện tích của từng box để ưu tiên box lớn
        areas = [np.abs((np.max(box, axis=0)[0] - np.min(box, axis=0)[0]) *
                        (np.max(box, axis=0)[1] - np.min(box, axis=0)[1])) for box in boxes]

        # Sắp xếp box theo diện tích giảm dần
        sorted_indices = np.argsort(areas)[::-1]
        boxes = [boxes[i] for i in sorted_indices]

        used = [False] * len(boxes)
        for i, box1 in enumerate(boxes):
            if used[i]:
                continue
            keep = True
            for j, box2 in enumerate(boxes):
                if i == j or used[j]:
                    continue
                iou = calculate_iou(box1, box2)
                if iou > iou_threshold:
                    # Nếu box trùng, ưu tiên giữ box lớn hơn (đã sắp xếp theo diện tích)
                    keep = False
                    break
                if is_box_inside(box2, box1, inside_threshold):
                    # Nếu box2 nằm trong box1, bỏ box2
                    used[j] = True
            if keep:
                filtered_boxes.append(box1)
                used[i] = True

        if not filtered_boxes:
            print("Không có box nào hợp lệ sau khi lọc")
            return None

        # Chọn box tốt nhất (lớn nhất sau khi lọc)
        best_box = filtered_boxes[0]
        # print(f"Chọn box tốt nhất trong {len(boxes)} box, còn {len(filtered_boxes)} box sau lọc")
        cropped_img = crop_and_straighten_image(image, best_box)

        return cropped_img

    def recognize_text(self, cropped_imgs):
        """Nhận diện văn bản trên các ảnh đã crop, sử dụng mô hình ONNX đã tải sẵn."""
        rec_results = []
        for img in cropped_imgs:
            if img is None:
                rec_results.append({"text": None})
                continue
            # Convert image to bytes
            _, img_data = cv2.imencode('.png', img)
            data = {"image": img_data.tobytes()}
            batch = transform(data, self.rec_ops)
            
            # Prepare for ONNX model
            images = np.expand_dims(batch[0], axis=0).astype(np.float32)
            
            # Run recognition with ONNX
            preds = self.rec_session.run([self.rec_output_name], {self.rec_input_name: images})[0]
            post_result = self.rec_post_process([preds])
            
            # Process recognition result
            info = None
            if isinstance(post_result, dict):
                rec_info = {}
                for key in post_result:
                    if len(post_result[key][0]) >= 2:
                        rec_info[key] = {
                            "label": post_result[key][0][0],
                            "score": float(post_result[key][0][1]),
                        }
                info = rec_info
            elif isinstance(post_result, list) and isinstance(post_result[0], int):
                info = {"label": str(post_result[0]), "score": 1.0}
            else:
                if len(post_result[0]) >= 2:
                    info = {"label": post_result[0][0], "score": float(post_result[0][1])}
            
            if info:
                rec_results.append({"text": info})
            else:
                rec_results.append({"text": None})
        return rec_results

def crop_and_straighten_image(img, box):
    """Crop image using box and straighten it to horizontal."""
    cropped_img = get_rotate_crop_image(img, np.array(box).astype(np.float32))
    return cropped_img

def draw_det_res(dt_boxes, img):
    """Draw detection boxes on the image."""
    src_im = img.copy()
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im

