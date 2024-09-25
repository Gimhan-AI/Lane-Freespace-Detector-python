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

def find_free_space_center(da_binary):
    free_space_points = np.column_stack(np.where(da_binary > 100))
    if free_space_points.size > 0:
        free_space_center = np.mean(free_space_points, axis=0)
        free_space_memory.append(free_space_center)
    else:
        free_space_memory.append(None)

    smoothed_free_space_center = smooth_free_space_center()

    return smoothed_free_space_center

def smooth_free_space_center():
    centers = [center for center in free_space_memory if center is not None]
    smoothed_center = np.mean(centers, axis=0) if centers else None

    return smoothed_center

def find_side_points(da_binary, free_space_center):
    free_space_points = np.column_stack(np.where(da_binary > 100))
    if free_space_points.size > 0 and free_space_center is not None:
        distances = np.linalg.norm(free_space_points - free_space_center, axis=1)
        sorted_indices = np.argsort(distances)

        left_point = None
        right_point = None

        for idx in sorted_indices:
            point = free_space_points[idx]
            if point[1] < free_space_center[1] and left_point is None:
                left_point = point
            elif point[1] > free_space_center[1] and right_point is None:
                right_point = point
            if left_point is not None and right_point is not None:
                break
    else:
        left_point = right_point = None

    return left_point, right_point

def draw_free_space_info(image, free_space_center, left_point, right_point):
    if free_space_center is not None:
        cv2.circle(image, (int(free_space_center[1]), int(free_space_center[0])), 5, (0, 255, 0), -1)

    if left_point is not None:
        cv2.circle(image, (int([1]), int(left_point[0])), 5, (0, 0, 255), -1)
    if right_point is not None:
        cv2.circle(image, (int(right_point[1]), int(right_point[0])), 5, (255, 0, 0), -1)

    cv2.arrowedLine(image, (0, image.shape[0]), (0, 0), (255, 255, 255), 2, tipLength=0.05)
    cv2.arrowedLine(image, (0, image.shape[0]), (image.shape[1], image.shape[0]), (255, 255, 255), 2, tipLength=0.05)

    cv2.putText(image, "X", (image.shape[1] - 20, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, "Y", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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

MEMORY_SIZE = 5
free_space_memory = collections.deque(maxlen=MEMORY_SIZE)

try:
    while True:
        ret, frame = camera_feed.read()
        if not ret:
            print("Failed to capture frame from camera feed.")
            break

        processed_frame, lane_binary, DA = Run(model, frame)

        free_space_center = find_free_space_center(DA)

        left_point, right_point = find_side_points(DA, free_space_center)

        processed_frame = draw_free_space_info(processed_frame, free_space_center, left_point, right_point)

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
