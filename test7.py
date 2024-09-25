import cv2
import numpy as np
from models.velloai_models import LaneFreeSpaceDetector as net
from yoloDet import YoloTRT
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
    return img_rs, LL

def initialize_tracking_points(lane_binary):
    points = []
    for y in range(lane_binary.shape[0]):
        for x in range(lane_binary.shape[1] // 2):
            if lane_binary[y, x] > 100:
                points.append((x, y))
    if points:
        points = np.array(points, dtype=np.float32)
        center_point = np.mean(points, axis=0).reshape(-1, 1, 2)
        return center_point
    return None

model = net.TwinLiteNet()
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load('models/velloai_models/lanefreespacemodel.pth'))
model.eval()

camera_feed = cv2.VideoCapture('videos/video0.mp4')

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

if not camera_feed.isOpened():
    print("Error: Could not open camera feed.")
    exit()

ret, frame = camera_feed.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
processed_frame, lane_binary = Run(model, frame)
p0 = initialize_tracking_points(lane_binary)  # Initialize tracking points

mask = np.zeros_like(frame)  # Create mask for drawing

while ret:
    ret, frame = camera_feed.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Update the center point based on the flow
            if good_new.size > 0:
                new_center = good_new.mean(axis=0)
                if not np.isnan(new_center).any():
                    mask = cv2.circle(mask, (int(new_center[0]), int(new_center[1])), 5, (255, 255, 255), -1)
                else:
                    print("Computed center is NaN, skipping drawing.")
            else:
                print("No valid tracking points in good_new to calculate.")
        else:
            print("Failed to calculate optical flow; p1 is None.")

        img = cv2.add(frame, mask)
        cv2.imshow('Left Lane Center Tracking', img)

        old_gray = frame_gray.copy()
        if p1 is not None and st.sum() > 0:  # Update p0 only if there are good points to track
            p0 = good_new.reshape(-1, 1, 2)
    else:
        print("No initial tracking points; check initialization of p0.")

    if cv2.waitKey(1) & 0xFF == 27:
        break


camera_feed.release()
cv2.destroyAllWindows()