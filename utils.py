
import json
import logging
import time

import cv2
import torch
import torchvision

# Thiết lập logging
import logging
import logging.handlers

logger = logging.getLogger('Number_recognition')

def setup_logger(
    logger_name='Number_recognition',
    log_file='output/Number_recognition.log',
    level=logging.DEBUG,
    max_bytes=10*1024*1024,
    backup_count=5,
    log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
):
    """
    Thiết lập logger với console và file handler (rotation).
    
    Args:
        logger_name (str): Tên của logger (mặc định: 'yolov5').
        log_file (str): Đường dẫn file log (mặc định: 'rtsp.log').
        level (int): Mức độ log (mặc định: logging.DEBUG).
        max_bytes (int): Kích thước tối đa của file log (mặc định: 10MB).
        backup_count (int): Số file backup tối đa (mặc định: 5).
        log_format (str): Định dạng log (mặc định: thời gian, tên, mức độ, thông điệp).
    
    Returns:
        logging.Logger: Đối tượng logger đã được cấu hình.
    """
    # Tạo logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Ngăn logger lan truyền lên các logger cấp cao hơn
    logger.propagate = False
    
    # Định dạng log
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def draw_results(frame, plate_tracks, number_tracks, track_text, track_plate, last_plate_number, fps_display, red_line, blue_line, colors):
    """
    Vẽ kết quả lên frame: bounding box, text, FPS, và đường đỏ/xanh.
    
    Args:
        frame (np.ndarray): Frame hình ảnh để vẽ.
        plate_tracks (list): Danh sách track của biển số.
        number_tracks (list): Danh sách track của số.
        track_text (dict): Trạng thái text của các track number.
        track_plate (dict): Trạng thái của các track plate.
        fps_display (float): FPS để hiển thị.
        red_line (tuple): Tọa độ đường đỏ (x1, y1, x2, y2).
        blue_line (tuple): Tọa độ đường xanh (x1, y1, x2, y2).
        colors (list): Danh sách màu cho các class.
    
    Returns:
        np.ndarray: Frame đã vẽ.
    """
    # Vẽ FPS
    cv2.putText(frame, f'FPS: {fps_display:.2f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Vẽ đường đỏ và xanh
    cv2.line(frame, (red_line[0], red_line[1]), (red_line[2], red_line[3]), (0, 0, 255), 2)
    cv2.line(frame, (blue_line[0], blue_line[1]), (blue_line[2], blue_line[3]), (0, 255, 0), 2)
    
    # Vẽ number tracks
    for track in number_tracks:
        tlbr = track.tlbr
        track_id = track.track_id
        cls = track.cls
        number_box = (int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3]))
        
        cv2.rectangle(frame, (number_box[0], number_box[1]), 
                      (number_box[2], number_box[3]), colors[cls], 2)
        if track_id in track_text and track_text[track_id]["recognized"]:
            cv2.putText(frame, f'Number: {track_text[track_id]["text"]}', 
                        (number_box[2], number_box[3] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    # Vẽ plate tracks
    for track in plate_tracks:
        tlbr = track.tlbr
        track_id = track.track_id
        cls = track.cls
        plate_box = (int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3]))
        
        cv2.rectangle(frame, (plate_box[0], plate_box[1]), 
                      (plate_box[2], plate_box[3]), colors[cls], 2)
        if track_id in track_plate:
            cv2.putText(frame, f'Plate: {last_plate_number}', 
                        (plate_box[2], plate_box[3] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    return frame

def rescale(frame, img_size, x_min, y_min, x_max, y_max):
    scale_x = frame.shape[1] / img_size
    scale_y = frame.shape[0] / img_size
    return x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y
def is_crossing_line(box, line):
    """Kiểm tra xem hộp giới hạn có cắt qua đường màu đỏ không"""
    x_min, y_min, x_max, y_max = box
    line_x1, line_y1, line_x2, line_y2 = line
    return x_min <= line_x1 <= x_max and y_min <= line_y2
def rotate_image(image, angle, scale=1.0):
    # Lấy kích thước ảnh
    height, width = image.shape[:2]
    # Tính tâm xoay (giữa ảnh)
    center = (width // 2, height // 2)
    # Tạo ma trận xoay
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    # Thực hiện xoay ảnh
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image
def crop_license_plate(frame, box):
    """Cắt vùng biển số"""
    x_min, y_min, x_max, y_max = box
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame.shape[1], x_max)
    y_max = min(frame.shape[0], y_max)
    cropped_img = frame[y_min:y_max, x_min:x_max]
    cropped_img = rotate_image(cropped_img, -40)
    return cropped_img

def save_to_json(last_plate_number, filename="output/plate_data.json"):
    data = {
        "last_plate_number": last_plate_number,
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def NumberFilter(number, N, state, delete_flag):
    """Hàm đánh giá số number và trả về cờ dừng cùng trạng thái cập nhật"""
    
    # Khởi tạo stop_flag với giá trị mặc định
    stop_flag = False
    number = int(number)

    if len(N) < 2:
        if state == 0:  # Trạng thái 0: Gán N0
            stop_flag = number % 5 == 0
            N.append(number)
            state = 1 if stop_flag else 0  # Cập nhật state trước khi append result
        elif state == 1:  # Trạng thái 1: Gán N1
            stop_flag = True
            if number % 5 == 0:
                if number == N[0] + 5:
                    N.append(number)
                    state = 2
                else:
                    delete_flag += 1
                    stop_flag = False
                    logger.warning(f"Number {number} may be incorrect at state 1 ")
                    if delete_flag > 1:
                        N.append(number)
                        state = 2
                        delete_flag = 0
                        stop_flag = True
            else:
                stop_flag = False
                logger.warning(f"Number {number} is not divisibel by 5 at state 1 ")
    else:
        if state == 2:  # Trạng thái 2: Gán N2        
            stop_flag = True
            if number % 5 == 0:
                if number == N[0] + 10 and number == N[1] + 5:
                    N.append(number)
                    N.pop(0)
                elif number == N[0] + 10:
                    N.append(number)
                    N.pop(0)
                    N.pop(1)
                    state = 1
                elif number == N[1] + 5:
                    N.append(number)
                    N.pop(0)
                else:
                    delete_flag += 1
                    stop_flag = False
                    logger.warning(f"Number {number} may be incorrect at state 2 ")
                    if delete_flag > 1:
                        N.append(number)
                        N.pop(0)
                        N.pop(1)
                        state = 1
                        delete_flag = 0
                        stop_flag = True
            else:
                logger.warning(f"Number {number} is not divisibel by 5 at state 2 ")
                stop_flag = False
                
    return stop_flag, N, state, delete_flag

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            logger.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output

