import torch
import numpy as np
import cv2
from models.velloai_models import LaneFreeSpaceDetector as net
from yoloDet import YoloTRT
import collections

def Run(model, img):
    # Resize and process image as needed by the model
    img = cv2.resize(img, (640, 480))  # Ensure this matches expected size
    img_rs = img.copy()

    # Model expects input in a different format, ensure this conversion doesn't alter size
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

    DA = da_predict.byte().cpu().data.numpy()[0] * 255

    # Detect edges of the free space using Canny edge detector
    edges = cv2.Canny(DA, 100, 200)

    # Detect lane lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    # Visualize the detected free space
    img_rs[DA > 100] = [0, 255, 0]  # Free space in green

    # Draw the detected lane lines on the processed image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_rs, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Lane lines in red

    return img_rs, edges, DA  # Return processed image, edges, and free space

def find_free_space_center(da_binary):
    # Find free space points
    free_space_points = np.column_stack(np.where(da_binary > 100))

    # Calculate the center of free space
    if free_space_points.size > 0:
        free_space_center = np.mean(free_space_points, axis=0)
        free_space_memory.append(free_space_center)
    else:
        free_space_memory.append(None)

    # Use memory to smooth free space center positions
    smoothed_free_space_center = smooth_free_space_center()

    return smoothed_free_space_center

def smooth_free_space_center():
    # Extract free space centers from memory buffer
    centers = [center for center in free_space_memory if center is not None]

    # Compute smoothed positions using the average of stored points
    smoothed_center = np.mean(centers, axis=0) if centers else None

    return smoothed_center

def draw_free_space_center_line(image, free_space_center, distance_forward=50):
    # Draw the center line based on free space
    if free_space_center is not None:
        # Draw the current free space center
        cv2.circle(image, (int(free_space_center[1]), int(free_space_center[0])), 5, (0, 255, 0), -1)  # Free space center in green

        # Calculate the point in front of the free space center
        new_point_y = free_space_center[0] - distance_forward  # Move `distance_forward` pixels upwards
        new_point_x = free_space_center[1]  # Keep the same x-coordinate
        
        # Ensure the new point is within image bounds
        new_point_y = max(0, new_point_y)  # Prevent y from going out of the top boundary
        new_point = (int(new_point_x), int(new_point_y))

        # Draw the new point in front of the free space center
        cv2.circle(image, new_point, 5, (255, 0, 0), -1)  # New point in blue

        # Optionally, draw a line between the free space center and the new point
        cv2.line(image, (int(free_space_center[1]), int(free_space_center[0])), new_point, (255, 255, 0), 2)  # Line in cyan

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

# Parameters for memory buffer
MEMORY_SIZE = 5  # Number of frames to keep in memory
free_space_memory = collections.deque(maxlen=MEMORY_SIZE)  # Memory buffer for free space points

try:
    while True:
        ret, frame = camera_feed.read()
        if not ret:
            print("Failed to capture frame from camera feed.")
            break

        # Get processed frame, lane prediction, and free space
        processed_frame, lane_edges, DA = Run(model, frame)

        # Find free space center
        free_space_center = find_free_space_center(DA)

        # Draw center line and new point in front based on free space
        processed_frame = draw_free_space_center_line(processed_frame, free_space_center, distance_forward=50)

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
