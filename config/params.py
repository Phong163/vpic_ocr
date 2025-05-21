# Thiết lập tham số
img_size = 480
red_line = (345, 0, 345, 190)  # (x1, y1, x2, y2)
blue_line = (550, 0, 550, 190)
conf_thres = 0.5 # Ngưỡng độ tin cậy
iou_thres = 0.4   # Ngưỡng IOU cho NMS
max_det = 1000    # Số lượng tối đa đối tượng phát hiện

# Define class names and colors for visualization
class_names = ['plate', 'number']
colors = [(0, 255, 0), (0, 0, 255)]  # Green for plate, Red for number


# Lưu trữ trạng thái nhận diện số cho từng track
track_plate = {}
track_text = {}  
first_number = 0  
last_plate_number = 0 
count = 0
current_number = 0  # Số hiện tại để gán cho plate_tracks


# Khởi tạo biến cho NumberFilter
N = []  # N0, N1, N2
state = 0               # Trạng thái hiện tại (0, 1, 2)
number_count = 0        # Đếm số plate có số
delete_flag = 0

#
video_writer = None
