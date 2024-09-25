import torch
import numpy as np
import cv2
from models.velloai_models import LaneFreeSpaceDetector as net
from yoloDet import YoloTRT

import numpy as np
import cv2
import torch

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

    # New code to find the center of the left lane
    if np.any(LL > 100):
        cols = np.where(LL.max(axis=0) > 100)[0]
        if len(cols) > 1:
            left_lane_center = cols[len(cols) // 4]  # Approximate center of the left quarter
            cv2.line(img_rs, (left_lane_center, 0), (left_lane_center, img_rs.shape[0]), (255, 255, 255), 2)
    
    return img_rs

def draw_axes(img):
    # Draw X-axis (horizontal line across the middle)
    cv2.line(img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), (255, 0, 0), 2)
    # Draw Y-axis (vertical line across the middle)
    cv2.line(img, (img.shape[1] // 2, 0), (img.shape[1] // 2, img.shape[0]), (0, 255, 0), 2)
    return img

model = net.TwinLiteNet()
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load('models/velloai_models/lanefreespacemodel.pth'))
model.eval()

model_yolo = YoloTRT(library="/home/vegaai/Lane-Freespace-Detector-python/models/velloai_models/yolov5/build/libmyplugins.so", engine="/home/vegaai/Lane-Freespace-Detector-python/models/velloai_models/yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")

video_feed = cv2.VideoCapture('videos/video0.mp4')

video_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not video_feed.isOpened():
    print("Error: Could not open video feed.")
    exit()

try:
    frame_count = 0
    while True:
        ret, frame = video_feed.read()
        if not ret:
            print("End of video stream or failed to read frame.")
            break

        frame_count += 1
        print(f"Processing frame {frame_count}")

        processed_frame = Run(model, frame)

        # processed_frame_with_axes = draw_axes(processed_frame)

        detections, t = model_yolo.Inference(processed_frame)

        print("FPS: {} sec".format(1/t))

        cv2.imshow('Processed Video Feed', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User requested exit.")
            break

except KeyboardInterrupt:
    print("\nRecording stopped by user.")

finally:
    video_feed.release()
    cv2.destroyAllWindows()
