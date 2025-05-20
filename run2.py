import time
import cv2
import torch
import numpy as np
import onnxruntime as ort
from tracker.tracker import BYTETracker
from utils import *
from config.params import *
import os
import argparse
from rstp_stream import RTSPStream
from e2e import OCRProcessor  

# Thiết lập logging
logger = setup_logger()

# Thiết lập biến môi trường để xử lý lỗi OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Hàm parse để xử lý các đường dẫn và URL RTSP
def parse_args():
    parser = argparse.ArgumentParser(description='License Plate Detection and OCR with RTSP')
    parser.add_argument('--rtsp_url', type=str, default='rtsp://admin:cxview2021@115.74.225.99:554/live',
                        help='RTSP URL for video stream')
    parser.add_argument('--output_video_path', type=str, default=os.path.join(os.getcwd(), 'output', 'output2.mp4'),
                        help='Path to output video')
    parser.add_argument('--yolo_weights', type=str, default=os.path.join(os.getcwd(), 'weights', 'yolov5n_bienso.onnx'),
                        help='Path to YOLO ONNX weights')
    parser.add_argument('--det_config_path', type=str, default=os.path.join(os.getcwd(), 'config', 'paddlev5_det.yaml'),
                        help='Path to OCR detection config')
    parser.add_argument('--det_onnx_path', type=str, default=os.path.join(os.getcwd(), 'weights', 'paddlev5_det.onnx'),
                        help='Path to OCR detection ONNX model')
    parser.add_argument('--rec_config_path', type=str, default=os.path.join(os.getcwd(), 'config', 'rec_mv3_none_bilstm_ctc.yml'),
                        help='Path to OCR recognition config')
    parser.add_argument('--rec_onnx_path', type=str, default=os.path.join(os.getcwd(), 'weights', 'paddle_rec.onnx'),
                        help='Path to OCR recognition ONNX model')
    parser.add_argument('--cam_id', type=int, default=0,
                        help='Camera ID for identification')
    args = parser.parse_args()

    # Kiểm tra sự tồn tại của các tệp
    for path in [args.yolo_weights, args.det_config_path, args.det_onnx_path, 
                 args.rec_config_path, args.rec_onnx_path]:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            exit(1)

    # Tạo thư mục đầu ra nếu chưa tồn tại
    output_dir = os.path.dirname(args.output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return args


# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lấy các đường dẫn và URL từ hàm parse
args = parse_args()
rtsp_url = args.rtsp_url
output_video_path = args.output_video_path
yolo_weights = args.yolo_weights
det_config_path = args.det_config_path
det_onnx_path = args.det_onnx_path
rec_config_path = args.rec_config_path
rec_onnx_path = args.rec_onnx_path
cam_id = args.cam_id

# Load YOLO
yolo_session = ort.InferenceSession(yolo_weights)
input_name = yolo_session.get_inputs()[0].name
output_name = yolo_session.get_outputs()[0].name

# Load OCR
ocr_processor = OCRProcessor(det_config_path, det_onnx_path, rec_config_path, rec_onnx_path)

# Khởi tạo RTSP stream
rtsp_stream = RTSPStream(rtsp_url, cam_id)
rtsp_stream.start()

# Khởi tạo Tracker
tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, track_buffer=50, frame_rate=15)

def detect_yolo(frame):
    """Hàm phát hiện biển số bằng YOLOv5"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    frame_tensor = frame_tensor.to(device)

    input_tensor = frame_tensor.cpu().numpy()
    detected = yolo_session.run(None, {input_name: input_tensor})[0]
    detected = torch.from_numpy(detected).to(device)
    detected = non_max_suppression(detected, conf_thres, iou_thres, max_det=max_det)
    return detected

prev_time = time.time()
try:
    while rtsp_stream.running:
        ret, frame = rtsp_stream.get_frame()
        if not ret:
            logger.warning("No frame from RTSP, trying again...")
            time.sleep(0.1)
            continue

        count += 1
        frame_resized = cv2.resize(frame, (img_size, img_size))
        pred = detect_yolo(frame_resized)
        
        # Chuẩn bị dữ liệu cho tracker
        detections = []
        if pred is not None and len(pred):
            for det in pred[0]:
                x_min, y_min, x_max, y_max, conf, cls = det[:6]
                x_min_rescaled, y_min_rescaled, x_max_rescaled, y_max_rescaled = rescale(
                    frame, img_size, x_min, y_min, x_max, y_max
                )
                tlbr = [x_min_rescaled, y_min_rescaled, x_max_rescaled, y_max_rescaled]
                detections.append([tlbr[0], tlbr[1], tlbr[2], tlbr[3], conf.cpu().numpy(), int(cls.cpu().numpy())])
        detections = np.array(detections) if detections else np.empty((0, 6))
        tracks = tracker.update(detections)
        plate_tracks = [t for t in tracks if t.cls == 0]
        number_tracks = [t for t in tracks if t.cls == 1]

        for track in number_tracks:
            tlbr = track.tlbr
            track_id = track.track_id
            cls = track.cls
            number_box = (int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3]))

            if track_id not in track_text:
                track_text[track_id] = {
                    "text": "Unknown",
                    "recognized": False,
                    "stopped": False,
                }

            is_crossing_1 = is_crossing_line(number_box, red_line)
            if is_crossing_1 and not track_text[track_id]["stopped"]:
                cropped_img = crop_license_plate(frame, number_box)
                cropped_img = ocr_processor.detect_text(cropped_img)
                rec_results = ocr_processor.recognize_text(cropped_img)
                if rec_results and rec_results[0]["text"] and rec_results[0]["text"].get("label"):
                    text = rec_results[0]["text"]["label"] 
                # if rec_results and rec_results[0]["text"] is not None:
                #     text = rec_results[0]["text"]  # Truy cập trực tiếp chuỗi
                    track_text[track_id]["text"] = text
                    last_plate_number = int(text)
                    stop_flag, N, state, delete_flag = NumberFilter(text, N, state, delete_flag)
                    track_text[track_id]["stopped"] = stop_flag
                    track_text[track_id]["recognized"] = True
                else:
                    track_text[track_id]["text"] = "Unrecognized"
                    track_text[track_id]["recognized"] = False
                    logger.warning(f"Not recognized for track_id {track_id}")

        for track in plate_tracks:
            tlbr = track.tlbr
            track_id = track.track_id
            cls = track.cls
            plate_box = (int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3]))

            if track_id not in track_plate:
                track_plate[track_id] = {
                    "stopped": False,
                }

            is_crossing_2 = is_crossing_line(plate_box, blue_line)
            if is_crossing_2 and not track_plate[track_id]["stopped"]:
                if number_tracks:
                    current_number = last_plate_number
                    track_plate[track_id]["stopped"] = True
                else:
                    current_number = last_plate_number + 1
                    last_plate_number = current_number
                    track_plate[track_id]["stopped"] = True
                save_to_json(last_plate_number)
        
        # Tính FPS
        curr_time = time.time()
        exec_time = curr_time - prev_time
        fps_display = 1 / exec_time if exec_time > 0 else 0
        prev_time = curr_time

        # Vẽ kết quả lên frame
        frame = draw_results(frame, plate_tracks, number_tracks, track_text, track_plate, last_plate_number,
                            fps_display, red_line, blue_line, colors)

        # Hiển thị frame
        frame = cv2.resize(frame, (1280, 640))

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    logger.info("Stopped by user.")
finally:
    rtsp_stream.stop()
    cv2.destroyAllWindows()
    logger.info("Released resources.")