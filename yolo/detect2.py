import torch
import pathlib
from ultralytics.utils.plotting import save_one_box
from yolo.utils.dataloaders import LoadImages
import matplotlib.pyplot as plt
# Sửa đổi pathlib để tương thích với Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from .utils.general import (
    Profile,
    check_img_size,
    non_max_suppression,
    scale_boxes
)

def run(source, model, device):
    #
    augment=False  # augmented inference
    visualize=False # visualize features
    vid_stride=1
    imgsz = (640, 640)
    agnostic_nms=False
    conf_thres=float(0.25)  # confidence threshold
    iou_thres=float(0.45)  # NMS IOU threshold
    max_det=int(1000)
    classes=None
    #
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    bs = 1
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)
        
        with dt[1]:
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
            
            

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        crop_yolo= []
        for i, det in enumerate(pred):  # per image
            
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
            s = "%gx%g " % im.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    crop = save_one_box(xyxy,im0,BGR=True)
                    crop_yolo.append((crop,xyxy))
                    
    return im0, crop_yolo


    
    
    
    
