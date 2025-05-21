import logging
import os
import sys
import cv2
import numpy as np
import yaml
import onnxruntime as ort
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process

# Setup logging and paths
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([__dir__, os.path.abspath(os.path.join(__dir__, ".."))])
logger = logging.getLogger('Number_recognition')

class OCRProcessor:
    def __init__(self, det_config_path, det_onnx_path, rec_config_path, rec_onnx_path):
        """Initialize detection and recognition models with ONNX."""
        if not all(os.path.exists(path) for path in [det_config_path, det_onnx_path, rec_config_path, rec_onnx_path]):
            raise FileNotFoundError("One or more model/config files not found")

        # Load configurations
        with open(det_config_path, 'r') as f:
            self.det_config = yaml.safe_load(f)
        with open(rec_config_path, 'r') as f:
            self.rec_config = yaml.safe_load(f)
        self.rec_global_config = self.rec_config["Global"]

        # Initialize detection
        self.det_session = ort.InferenceSession(det_onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.det_output_name = self.det_session.get_outputs()[0].name
        self.det_post_process = build_post_process(self.det_config["PostProcess"])
        self.det_ops = create_operators(self._setup_transforms(self.det_config["Eval"]["dataset"]["transforms"], 
                                                              keep_keys=["image", "shape"]))

        # Initialize recognition
        self.rec_session = ort.InferenceSession(rec_onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        self.rec_output_name = self.rec_session.get_outputs()[0].name
        self.rec_post_process = build_post_process(self.rec_config["PostProcess"], self.rec_global_config)
        self.rec_ops = create_operators(self._setup_transforms(self.rec_config["Eval"]["dataset"]["transforms"], 
                                                              keep_keys=["image"], infer_mode=True))
        self.rec_global_config["infer_mode"] = True

    def _setup_transforms(self, transforms, keep_keys=None, infer_mode=False):
        """Helper to setup transforms for detection or recognition."""
        result = []
        for op in transforms:
            op_name = list(op)[0]
            if "Label" in op_name:
                continue
            if op_name == "KeepKeys" and keep_keys:
                op[op_name]["keep_keys"] = keep_keys
            if op_name == "RecResizeImg" and infer_mode:
                op[op_name]["infer_mode"] = True
            result.append(op)
        return result

    def detect_text(self, image):
        """Detect text in the image using pre-loaded ONNX model."""
        if not isinstance(image, np.ndarray) or image.size == 0:
            logger.error("Invalid or empty input image")
            return None

        image = cv2.resize(image, (960, 736))
        _, img_data = cv2.imencode('.png', image)
        data = {"image": img_data.tobytes()}
        batch = transform(data, self.det_ops)

        # Run detection
        images = np.expand_dims(batch[0], axis=0).astype(np.float32)
        shape_list = np.expand_dims(batch[1], axis=0)
        preds = self.det_session.run([self.det_output_name], {self.det_input_name: images})[0]
        post_result = self.det_post_process({"maps": preds}, shape_list)

        # Process detection results
        boxes = post_result.get(list(post_result.keys())[0], [{}])[0].get("points", []) if isinstance(post_result, dict) else post_result[0]["points"]
        if not boxes:
            logger.warning("No text boxes detected")
            return None

        return crop_and_straighten_image(image, boxes[0])

    def recognize_text(self, cropped_img):
        """Recognize text in cropped image using pre-loaded ONNX model."""
        if not isinstance(cropped_img, np.ndarray) or cropped_img.size == 0:
            logger.error("Invalid or empty cropped image")
            return [{"text": None}]

        try:
            _, img_data = cv2.imencode('.png', cropped_img)
        except cv2.error as e:
            logger.error(f"Image encoding error: {e}")
            return [{"text": None}]

        data = {"image": img_data.tobytes()}
        batch = transform(data, self.rec_ops)
        images = np.expand_dims(batch[0], axis=0).astype(np.float32)
        preds = self.rec_session.run([self.rec_output_name], {self.rec_input_name: images})[0]
        post_result = self.rec_post_process([preds])

        # Process recognition results
        if isinstance(post_result, dict):
            info = {key: {"label": val[0][0], "score": float(val[0][1])} for key, val in post_result.items() if len(val[0]) >= 2}
        elif isinstance(post_result, list) and isinstance(post_result[0], int):
            info = {"label": str(post_result[0]), "score": 1.0}
        else:
            info = {"label": post_result[0][0], "score": float(post_result[0][1])} if len(post_result[0]) >= 2 else None

        return [{"text": info}] if info else [{"text": None}]

def crop_and_straighten_image(img, points):
    """Crop and straighten image based on detected box."""
    if len(points) != 4:
        logger.error("Invalid box points: must be 4*2")
        return None

    points = np.array(points, dtype=np.float32)
    width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    M = cv2.getPerspectiveTransform(points, dst_pts)
    dst_img = cv2.warpPerspective(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 1, 1.0)
    return cv2.warpAffine(dst_img, rotation_matrix, (width, height))

def draw_det_res(dt_boxes, img):
    """Draw detection boxes on the image."""
    if not isinstance(img, np.ndarray) or not dt_boxes:
        logger.error("Invalid image or empty detection boxes")
        return img

    src_im = img.copy()
    for box in dt_boxes:
        box = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im