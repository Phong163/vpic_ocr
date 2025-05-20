import logging
import os
import sys
import cv2
import numpy as np
import yaml
import onnxruntime as ort
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process

# Add PaddleOCR to system path
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

# Setup logging
logger = logging.getLogger('Number_recognition')

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
        image = cv2.resize(image, (960, 736))  # Thay 960x960 bằng kích thước mong muốn
        # Convert image to bytes for process    ing
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
            return None
        best_box = boxes[0]
        cropped_img = crop_and_straighten_image(image, best_box)
        return cropped_img
    
    def recognize_text(self, cropped_imgs):
        """Nhận diện văn bản trên ảnh đã crop, sử dụng mô hình ONNX đã tải sẵn."""
        rec_results = []

        # Kiểm tra xem cropped_imgs có hợp lệ không
        if cropped_imgs is None or not isinstance(cropped_imgs, np.ndarray) or cropped_imgs.size == 0:
            logger.error("Hinh anh dau vao khong hop le hoac rong trong recognize_text")
            return [{"text": None}]

        # Chuyển đổi hình ảnh sang bytes
        try:
            _, img_data = cv2.imencode('.png', cropped_imgs)
        except cv2.error as e:
            logger.error(f"Loi khi ma hoa hinh anh: {e}")
            return [{"text": None}]

        data = {"image": img_data.tobytes()}
        batch = transform(data, self.rec_ops)
        
        # Chuẩn bị cho mô hình ONNX
        images = np.expand_dims(batch[0], axis=0).astype(np.float32)
        
        # Chạy nhận diện với ONNX
        preds = self.rec_session.run([self.rec_output_name], {self.rec_input_name: images})[0]
        post_result = self.rec_post_process([preds])
        
        # Xử lý kết quả nhận diện
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

def get_rotate_crop_image(img, points):
    assert len(points) == 4, "shape of points must be 4*2"
    
    # Tính kích thước ảnh cắt dựa trên các điểm
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    
    # Định nghĩa các điểm chuẩn
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    
    # Tạo ma trận biến đổi perspective
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    
    # Xoay ảnh 30 độ quanh tâm
    center = (img_crop_width // 2, img_crop_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 1, 1.0)
    dst_img = cv2.warpAffine(dst_img, rotation_matrix, (img_crop_width, img_crop_height))
    
    return dst_img

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


