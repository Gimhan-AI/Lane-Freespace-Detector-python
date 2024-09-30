import torch
import numpy as np
import cv2
from models.velloai_models import LaneFreeSpaceDetector as net
from yoloDet import YoloTRT

prev_left_furthest = None
prev_right_furthest = None

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

def find_free_space_edge_points(free_space_binary):
    contours, _ = cv2.findContours(free_space_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_points = []
    for contour in contours:
        for point in contour:
            edge_points.append(tuple(point[0]))
    if edge_points:
        closest_point = min(edge_points, key=lambda point: point[1])
        return closest_point
    return None

def find_lane_furthest_points(lane_binary):
    global prev_left_furthest, prev_right_furthest
    edges = cv2.Canny(lane_binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    left_lane_points, right_lane_points = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 < lane_binary.shape[1] // 2 and x2 < lane_binary.shape[1] // 2:
                left_lane_points.append((x1, y1))
                left_lane_points.append((x2, y2))
            elif x1 >= lane_binary.shape[1] // 2 and x2 >= lane_binary.shape[1] // 2:
                right_lane_points.append((x1, y1))
                right_lane_points.append((x2, y2))
    left_furthest = max(left_lane_points, key=lambda point: point[1]) if left_lane_points else prev_left_furthest
    right_furthest = max(right_lane_points, key=lambda point: point[1]) if right_lane_points else prev_right_furthest
    prev_left_furthest = left_furthest
    prev_right_furthest = right_furthest
    return left_furthest, right_furthest

def draw_directional_arrow(image, start_point, end_point):
    if start_point and end_point:
        angle = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        arrow_length = 100  # Fixed size for the small arrow
        end_point_arrow = (int(start_point[0] + arrow_length * np.cos(angle)), int(start_point[1] + arrow_length * np.sin(angle)))
        cv2.arrowedLine(image, start_point, end_point_arrow, (255, 255, 0), 3, tipLength=0.3)
    # cv2.line(image, start_point, (350, 100), (0, 255, 255), 3)
    fixed_angle = np.arctan2(100 - 460, 350 - 350)  # This is 0 because the line is vertical
    steering_angle = np.degrees(angle - fixed_angle)
    steering_angle = np.clip(steering_angle, -45, 45)
    cv2.putText(image, f"Steering Angle: {steering_angle:.2f} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return image

model = net.TwinLiteNet()
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load('models/velloai_models/lanefreespacemodel.pth'))
model.eval()
model_yolo = YoloTRT(library="/home/vegaai/Lane-Freespace-Detector-python/models/velloai_models/yolov5/build/libmyplugins.so", engine="/home/vegaai/Lane-Freespace-Detector-python/models/velloai_models/yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")
# camera_feed = cv2.VideoCapture('videos/video0.mp4')
camera_feed = cv2.VideoCapture('videos/trace.webm')
camera_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not camera_feed.isOpened():
    print("Error: Could not open camera feed.")
    exit()

try:
    while True:
        ret, frame = camera_feed.read()
        if not ret:
            print("Failed to capture frame from camera feed.")
            break
        processed_frame, lane_binary, DA = Run(model, frame)
        closest_fs_point = find_free_space_edge_points(DA)
        start_point = (350, 460)  # Fixed point at the bottom center
        if closest_fs_point:
            processed_frame = draw_directional_arrow(processed_frame, start_point, closest_fs_point)
        cv2.imshow('Processed Camera Feed', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User requested exit.")
            break
except KeyboardInterrupt:
    print("\nProcessing stopped by user.")
finally:
    camera_feed.release()
    cv2.destroyAllWindows()
    print("Camera feed processing terminated.")
