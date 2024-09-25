import torch
import numpy as np
import cv2
from models.velloai_models import LaneFreeSpaceDetector as net
from yoloDet import YoloTRT
import collections

def Run(model, img):
    img = cv2.resize(img, (640, 480))  # Ensure this matches expected size
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

    # Visualize the detected free space and lanes
    img_rs[DA > 100] = [255, 0, 0]  # Free space in red
    img_rs[LL > 100] = [0, 255, 0]  # Lanes in green

    return img_rs, LL, DA  # Return processed image, lane prediction, and free space


def find_lane_centers(lane_binary):
    # Detect edges and lines
    edges = cv2.Canny(lane_binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    left_lane_points = []
    right_lane_points = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 < lane_binary.shape[1] // 2 and x2 < lane_binary.shape[1] // 2:  # Left lane
                left_lane_points.append((x1, y1))
                left_lane_points.append((x2, y2))
            elif x1 > lane_binary.shape[1] // 2 and x2 > lane_binary.shape[1] // 2:  # Right lane
                right_lane_points.append((x1, y1))
                right_lane_points.append((x2, y2))

    # Calculate the centers of the lanes
    left_lane_center = np.mean(left_lane_points, axis=0) if left_lane_points else None
    right_lane_center = np.mean(right_lane_points, axis=0) if right_lane_points else None

    # Append current lane center to memory buffer
    lane_memory.append((left_lane_center, right_lane_center))

    # Use memory to smooth lane center positions
    smoothed_left_center, smoothed_right_center = smooth_lane_centers()

    return smoothed_left_center, smoothed_right_center

def smooth_lane_centers():
    # Extract lane centers from memory buffer
    left_centers = [center[0] for center in lane_memory if center[0] is not None]
    right_centers = [center[1] for center in lane_memory if center[1] is not None]

    # Compute smoothed positions using average of stored points
    smoothed_left_center = np.mean(left_centers, axis=0) if left_centers else None
    smoothed_right_center = np.mean(right_centers, axis=0) if right_centers else None

    return smoothed_left_center, smoothed_right_center

def draw_center_line(image, left_lane_center, right_lane_center):
    # Calculate the center line
    if left_lane_center is not None and right_lane_center is not None:
        # Midpoint between left and right lane centers
        road_center_x = (left_lane_center[0] + right_lane_center[0]) / 2
        road_center_y = (left_lane_center[1] + right_lane_center[1]) / 2

        # Draw lane centers
        cv2.circle(image, (int(left_lane_center[0]), int(left_lane_center[1])), 5, (255, 0, 0), -1)  # Left lane center in blue
        cv2.circle(image, (int(right_lane_center[0]), int(right_lane_center[1])), 5, (0, 0, 255), -1)  # Right lane center in red

        # Draw the center line between lane centers
        cv2.line(image, (int(left_lane_center[0]), int(left_lane_center[1])), 
                 (int(right_lane_center[0]), int(right_lane_center[1])), (0, 255, 255), 2)  # Center line in yellow

        # Draw road center point
        cv2.circle(image, (int(road_center_x), int(road_center_y)), 5, (0, 255, 0), -1)  # Road center in green

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

# Initialize previous lane centers and road center
prev_left_lane_center = np.array([0, 0])
prev_right_lane_center = np.array([camera_feed.get(cv2.CAP_PROP_FRAME_WIDTH), 0])
prev_road_center = ((prev_left_lane_center[0] + prev_right_lane_center[0]) / 2,
                    (prev_left_lane_center[1] + prev_right_lane_center[1]) / 2)

# Parameters for memory buffer
MEMORY_SIZE = 5  # Number of frames to keep in memory
lane_memory = collections.deque(maxlen=MEMORY_SIZE)  # Memory buffer for lane points

try:
    frame_count = 0
    while True:
        ret, frame = camera_feed.read()
        if not ret:
            print("Failed to capture frame from camera feed.")
            break

        # Get processed frame, lane prediction, and free space
        processed_frame, lane_binary, DA = Run(model, frame)

        # Find lane centers
        left_lane_center, right_lane_center = find_lane_centers(lane_binary)

        # Draw center line between the lanes
        processed_frame = draw_center_line(processed_frame, left_lane_center, right_lane_center)

        # Display the processed camera feed
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
