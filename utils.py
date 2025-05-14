
import json

def rescale(frame, img_size, x_min, y_min, x_max, y_max):
    scale_x = frame.shape[1] / img_size
    scale_y = frame.shape[0] / img_size
    return x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y
def is_crossing_line(box, line):
    """Kiểm tra xem hộp giới hạn có cắt qua đường màu đỏ không"""
    x_min, y_min, x_max, y_max = box
    line_x1, line_y1, line_x2, line_y2 = line
    return x_min <= line_x1 <= x_max and y_min <= line_y2

def crop_license_plate(frame, box):
    """Cắt vùng biển số"""
    x_min, y_min, x_max, y_max = box
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame.shape[1], x_max)
    y_max = min(frame.shape[0], y_max)
    cropped_img = frame[y_min:y_max, x_min:x_max]
    return cropped_img

def find_nearest_plate(number_box, plate_tracks):
    """Tìm track plate gần nhất với track number dựa trên khoảng cách IoU"""
    number_tlbr = number_box
    min_dist = float('inf')
    nearest_plate_id = None
    
    for plate_track in plate_tracks:
        plate_tlbr = plate_track.tlbr
        iou = compute_iou(number_tlbr, plate_tlbr)
        if iou > 0 and iou < min_dist:
            min_dist = iou
            nearest_plate_id = plate_track.track_id
    
    return nearest_plate_id

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    ox1, oy1, ox2, oy2 = box2
    # Tính giao nhau
    xi1 = max(x1, ox1)
    yi1 = max(y1, oy1)
    xi2 = min(x2, ox2)
    yi2 = min(y2, oy2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    # Tính hợp
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (ox2 - ox1) * (oy2 - oy1)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def save_to_json(last_plate_number, filename="output/plate_data.json"):
    data = {
        "last_plate_number": last_plate_number,
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def NumberFilter(number, N, state, delete_flag):
    """Hàm đánh giá số number và trả về cờ dừng cùng trạng thái cập nhật"""
    global logger
    
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
                    if delete_flag > 2:
                        N.append(number)
                        state = 2
                        delete_flag = 0
                        stop_flag = True
            else:
                stop_flag = False
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
                    if delete_flag > 2:
                        N.append(number)
                        N.pop(0)
                        N.pop(1)
                        state = 1
                        delete_flag = 0
                        stop_flag = True
            else:
                stop_flag = False
        # Nếu state != 2, stop_flag giữ giá trị mặc định (False)

    return stop_flag, N, state, delete_flag

