import torch
import numpy as np
import cv2
from models.velloai_models import LaneFreeSpaceDetector as net
from yoloDet import YoloTRT
import collections

def Run(model, img):
    img = cv2.resize(img, (640, 480))
    img_rs = img.copy()
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)
    img = img.cuda().float() / 255.0
    with torch.no_grad():
        img_out = model(img)
    x0 = img_out[0]
    x1 = img_out[1]
    _, da_predict = torch.max(x0, 1)
    _, ll_predict = torch.max(x1, 1)
    DA = da_predict.byte().cpu().data.numpy()[0] * 255
    LL = ll_predict.byte().cpu().data.numpy()[0] * 255
    img_rs[DA > 100] = [255, 0, 0]
    img_rs[LL > 100] = [0, 255, 0]
    return img_rs, LL, DA

def find_left_lane_center(lane_binary):
    left_lane_points = []
    for y in range(lane_binary.shape[0]):
        for x in range(lane_binary.shape[1] // 2):
            if lane_binary[y, x] > 100:
                left_lane_points.append((x, y))
    left_lane_center = np.mean(left_lane_points, axis=0) if left_lane_points else None
    return left_lane_center

def draw_left_lane_center(image, left_lane_center):
    if left_lane_center is not None:
        cv2.circle(image, (int(left_lane_center[0]), int(left_lane_center[1])), 5, (255, 255, 255), -1)
    return image

model = net.TwinLiteNet()
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load('models/velloai_models/lanefreespacemodel.pth'))
model.eval()

model_yolo = YoloTRT(library="/home/vegaai/Lane-Freespace-Detector-python/models/velloai_models/yolov5/build/libmyplugins.so", engine="/home/vegaai/Lane-Freespace-Detector-python/models/velloai_models/yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")

camera_feed = cv2.VideoCapture('videos/video0.mp4')
camera_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera_feed.isOpened():
    print("Error: Could not open camera feed.")
    exit()

try:
    while True:
        ret, frame = camera_feed.read()
        if not ret:
            print("Failed to capture frame from camera feed or end of video.")
            break
        processed_frame, lane_binary, DA = Run(model, frame)
        left_lane_center = find_left_lane_center(lane_binary)
        if processed_frame is not None and left_lane_center is not None:
            processed_frame = draw_left_lane_center(processed_frame, left_lane_center)
            cv2.imshow('Processed Camera Feed', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User requested exit.")
            break

finally:
    camera_feed.release()
    cv2.destroyAllWindows()
    print("Camera feed processing terminated.")
