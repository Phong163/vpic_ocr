# Thiết lập tham số
img_size = 480
red_line = (345, 0, 345, 190)  # (x1, y1, x2, y2)
blue_line = (550, 0, 550, 190)
conf_thres = 0.5 # Ngưỡng độ tin cậy
iou_thres = 0.4   # Ngưỡng IOU cho NMS
max_det = 1000    # Số lượng tối đa đối tượng phát hiện
max_frames = 1
# Define class names and colors for visualization
class_names = ['plate', 'number']
colors = [(0, 255, 0), (0, 0, 255)]  # Green for plate, Red for number

# Lưu trữ trạng thái nhận diện số cho từng track
track_plate = {}
track_text = {}  # {track_id: {"text": str, "recognized": bool, "stopped": bool, "frame_count": int, "cls": int}}
first_number = 0  # Giá trị mặc định cho first_number
last_displayed = {"id": 0, "number": first_number}  # Trạng thái hiển thị gần nhất
number_to_plate = {}  # {number_track_id: plate_track_id} để ánh xạ number -> plate
last_plate_number = 0  # Số của plate gần nhất
c = 0
count = 0
current_number = 0  # Số hiện tại để gán cho plate_tracks

# Khởi tạo biến cho NumberFilter
N = []  # N0, N1, N2
accuracy = []    # Accuracy của N0, N1, N2
result = []             # Kết quả lưu trạng thái plates
state = 0               # Trạng thái hiện tại (0, 1, 2)
number_count = 0        # Đếm số plate có số
delete_flag = 0