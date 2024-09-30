import torch
import numpy as np
import cv2
from models.velloai_models import LaneFreeSpaceDetector as net
from yoloDet import YoloTRT

# Initialize the previous closest and furthest points to handle missing detections
prev_closest_point = None
prev_furthest_point = None

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
    _, ll_predict = torch.max(x1, 1)

    DA = da_predict.byte().cpu().data.numpy()[0] * 255
    LL = ll_predict.byte().cpu().data.numpy()[0] * 255

    # Visualize the detected free space and lanes
    img_rs[DA > 100] = [255, 0, 0]  # Free space in red
    img_rs[LL > 100] = [0, 255, 0]  # Lanes in green

    return img_rs, LL, DA  # Return processed image, lane prediction, and free space

def find_free_space_edge_points(free_space_binary):
    # Find contours in the binary image
    contours, _ = cv2.findContours(free_space_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_points = []

    for contour in contours:
        for point in contour:
            edge_points.append(tuple(point[0]))

    # Find the furthest and closest points based on y-coordinate
    if edge_points:
        furthest_point = max(edge_points, key=lambda point: point[1])
        closest_point = min(edge_points, key=lambda point: point[1])
        return closest_point, furthest_point
    return None, None

def draw_free_space_arrow(image, closest_point, furthest_point):
    # Draw an arrow pointing from the furthest point to the closest point
    if closest_point and furthest_point:
        cv2.arrowedLine(image, (int(furthest_point[0]), int(furthest_point[1])),
                        (int(closest_point[0]), int(closest_point[1])),
                        (0, 0, 255), 3, tipLength=0.3)  # Arrow in blue for free space
    return image

def find_left_lane_points(lane_binary):
    # Detect edges and lines
    edges = cv2.Canny(lane_binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    left_lane_points = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 < lane_binary.shape[1] // 2 and x2 < lane_binary.shape[1] // 2:  # Left lane
                left_lane_points.append((x1, y1))
                left_lane_points.append((x2, y2))

    # Find the furthest point (the point with the highest y-coordinate) and closest point (lowest y-coordinate)
    furthest_point = max(left_lane_points, key=lambda point: point[1]) if left_lane_points else None
    closest_point = min(left_lane_points, key=lambda point: point[1]) if left_lane_points else None

    return closest_point, furthest_point

def draw_arrow(image, closest_point, furthest_point):
    global prev_closest_point, prev_furthest_point
    
    # If the closest point and furthest point are not found, use the previous ones
    if closest_point is None or furthest_point is None:
        closest_point = prev_closest_point
        furthest_point = prev_furthest_point

    if closest_point is not None and furthest_point is not None:
        # Draw an arrow pointing from the furthest point to the closest point
        cv2.arrowedLine(image, (int(furthest_point[0]), int(furthest_point[1])),
                        (int(closest_point[0]), int(closest_point[1])),
                        (0, 255, 0), 3, tipLength=0.3)  # Arrow in green

        # Update previous points for future frames
        prev_closest_point = closest_point
        prev_furthest_point = furthest_point

    else:
        # If no points are available, draw a straight arrow
        img_center = (image.shape[1] // 2, image.shape[0])
        arrow_end = (image.shape[1] // 2, image.shape[0] // 2)
        cv2.arrowedLine(image, img_center, arrow_end, (0, 255, 0), 3, tipLength=0.3)  # Default straight arrow

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
    frame_count = 0
    while True:
        ret, frame = camera_feed.read()
        if not ret:
            print("Failed to capture frame from camera feed.")
            break

        # Get processed frame, lane prediction, and free space
        processed_frame, lane_binary, DA = Run(model, frame)

        # Find closest and furthest detection points for lanes
        closest_point, furthest_point = find_left_lane_points(lane_binary)
        processed_frame = draw_arrow(processed_frame, closest_point, furthest_point)

        # Find closest and furthest points for free space
        closest_fs_point, furthest_fs_point = find_free_space_edge_points(DA)
        processed_frame = draw_free_space_arrow(processed_frame, closest_fs_point, furthest_fs_point)

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

