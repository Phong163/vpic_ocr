import logging
import os
import cv2
import numpy as np
import yaml
import onnxruntime as ort
from ppocr.postprocess import build_post_process

logger = logging.getLogger('Number_recognition')


def preprocess_image(data, preprocess_type, config):
    if isinstance(data["image"], bytes):
        nparr = np.frombuffer(data["image"], np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        img = data["image"]

    if img is None or len(img.shape) != 3 or img.shape[2] != 3:
        logger.error("Ảnh đầu vào không hợp lệ hoặc không phải RGB")
        return None

    if preprocess_type == "detection":
        size = config.get("size", [960, 736])
        orig_height, orig_width = data.get("orig_height", img.shape[0]), data.get("orig_width", img.shape[1])
        img = cv2.resize(img, tuple(size))
        img = img.astype(np.float32)
        mean = np.array(config.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.array(config.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        img = img / 255.0
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        ratio_h = size[1] / orig_height
        ratio_w = size[0] / orig_width
        shape_list = np.array([[orig_height, orig_width, ratio_h, ratio_w]], dtype=np.float32)
        return img, shape_list

    elif preprocess_type == "recognition":
        height = config.get("height", 32)
        max_width = config.get("max_width", 100)
        logger.debug("Config height: %d, max_width: %d", height, max_width)  # Kiểm tra config
        if height != 32 or max_width != 100:
            logger.warning("Invalid config values: height=%d, max_width=%d. Forcing defaults.", height, max_width)
            height = 32
            max_width = 100

        h, w = img.shape[:2]
        logger.debug("Original image shape for recognition: (%d, %d)", h, w)
        
        # Tính tỷ lệ để đạt chiều cao mục tiêu
        scale = height / h
        new_width = min(int(w * scale), max_width)
        logger.debug("Resized dimensions: (%d, %d)", height, new_width)
        
        # Resize ảnh
        img = cv2.resize(img, (new_width, height))
        logger.debug("Shape after resize: %s", img.shape)
        
        # Đệm ảnh nếu cần
        if new_width < max_width:
            img_padded = np.zeros((height, max_width, 3), dtype=np.uint8)
            img_padded[:, :new_width, :] = img
            img = img_padded
        
        logger.debug("Padded image shape: %s", img.shape)
        
        # Chuẩn hóa ảnh
        img = img.astype(np.float32)
        mean = np.array(config.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.array(config.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        img = img / 255.0
        img = (img - mean) / std
        
        # Hoán vị: (height, width, channels) -> (channels, height, width)
        img = img.transpose(2, 0, 1)
        logger.debug("Shape after transpose: %s", img.shape)
        
        # Thêm chiều batch
        img = np.expand_dims(img, axis=0).astype(np.float32)
        logger.debug("Final tensor shape for recognition: %s", img.shape)
        
        return img

    else:
        raise ValueError("preprocess_type phải là 'detection' hoặc 'recognition'")
   
def load_preprocess_config(config, preprocess_type):
    params = {}
    if preprocess_type == "detection":
        transforms = config.get("Eval", {}).get("dataset", {}).get("transforms", [])
        for op in transforms:
            op_name = list(op)[0]
            op_params = op[op_name]
            if op_name == "Resize":
                params["size"] = op_params.get("size", [960, 736])
            elif op_name == "NormalizeImage":
                params["mean"] = op_params.get("mean", [0.485, 0.456, 0.406])
                params["std"] = op_params.get("std", [0.229, 0.224, 0.225])
    elif preprocess_type == "recognition":
        transforms = config.get("Eval", {}).get("dataset", {}).get("transforms", [])
        for op in transforms:
            op_name = list(op)[0]
            op_params = op[op_name]
            if op_name == "RecResizeImg":
                image_shape = op_params.get("image_shape", [32, 100])
                params["height"] = image_shape[1]  # Đảm bảo height = 32
                params["max_width"] = image_shape[2]  # Đảm bảo max_width = 100
            elif op_name == "NormalizeImage":
                params["mean"] = op_params.get("mean", [0.485, 0.456, 0.406])
                params["std"] = op_params.get("std", [0.229, 0.224, 0.225])
    logger.debug("Loaded config params for %s: %s", preprocess_type, params)  # Thêm log để kiểm tra
    return params

class OCRProcessor:
    def __init__(self, det_config_path, det_onnx_path, rec_config_path, rec_onnx_path):
        with open(det_config_path, 'r') as f:
            self.det_config = yaml.safe_load(f)
        self.det_session = ort.InferenceSession(det_onnx_path)
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.det_output_name = self.det_session.get_outputs()[0].name
        self.det_post_process = build_post_process(self.det_config["PostProcess"])
        self.det_config_params = load_preprocess_config(self.det_config, "detection")

        with open(rec_config_path, 'r') as f:
            self.rec_config = yaml.safe_load(f)
        self.rec_global_config = self.rec_config["Global"]
        self.rec_session = ort.InferenceSession(rec_onnx_path)
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        self.rec_output_name = self.rec_session.get_outputs()[0].name
        self.rec_post_process = build_post_process(self.rec_config["PostProcess"], self.rec_global_config)
        self.rec_config_params = load_preprocess_config(self.rec_config, "recognition")

    def detect_text(self, image):
        if image is None:
            logger.error("Ảnh đầu vào không hợp lệ")
            return None
        _, img_data = cv2.imencode('.png', image)
        data = {"image": img_data.tobytes(), "orig_height": image.shape[0], "orig_width": image.shape[1]}
        result = preprocess_image(data, "detection", self.det_config_params)
        if result is None:
            logger.error("Tiền xử lý thất bại")
            return None
        images, shape_list = result
        logger.debug("Detection tensor shape: %s, dtype: %s", images.shape, images.dtype)  # Kiểm tra kiểu dữ liệu
        preds = self.det_session.run([self.det_output_name], {self.det_input_name: images})[0]
        post_result = self.det_post_process({"maps": preds}, shape_list)
        if isinstance(post_result, dict):
            boxes = post_result[list(post_result.keys())[0]][0]["points"]
        else:
            boxes = post_result[0]["points"]
        if not len(boxes):
            logger.warning("Không phát hiện hộp nào")
            return None
        best_box = boxes[0]
        cropped_img = crop_and_straighten_image(image, best_box)
        return cropped_img

    def recognize_text(self, cropped_imgs):
        if cropped_imgs is None or not isinstance(cropped_imgs, np.ndarray) or cropped_imgs.size == 0:
            logger.error("Ảnh đầu vào không hợp lệ hoặc rỗng trong recognize_text")
            return [{"text": None}]
        _, img_data = cv2.imencode('.png', cropped_imgs)
        data = {"image": img_data.tobytes()}
        images = preprocess_image(data, "recognition", self.rec_config_params)
        if images is None:
            logger.error("Tiền xử lý thất bại")
            return [{"text": None}]
        logger.debug("Recognition tensor shape: %s, dtype: %s", images.shape, images.dtype)  # Kiểm tra kiểu dữ liệu
        preds = self.rec_session.run([self.rec_output_name], {self.rec_input_name: images})[0]
        post_result = self.rec_post_process([preds])
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
            return [{"text": info}]
        return [{"text": None}]

def get_rotate_crop_image(img, points):
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    center = (img_crop_width // 2, img_crop_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 1, 1.0)
    dst_img = cv2.warpAffine(dst_img, rotation_matrix, (img_crop_width, img_crop_height))
    return dst_img

def crop_and_straighten_image(img, box):
    return get_rotate_crop_image(img, np.array(box).astype(np.float32))

def draw_det_res(dt_boxes, img):
    src_im = img.copy()
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im