import torch
import numpy as np
import cv2
from models.velloai_models import LaneFreeSpaceDetector as net
from yoloDet import YoloTRT

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
    img_rs[DA > 100] = [255, 0, 0]
    img_rs[LL > 100] = [0, 255, 0]

    return img_rs

model = net.TwinLiteNet()
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load('models/velloai_models/lanefreespacemodel.pth'))
model.eval()

model_yolo = YoloTRT(library="/home/vegaai/Lane-Freespace-Detector-python/models/velloai_models/yolov5/build/libmyplugins.so", engine="/home/vegaai/Lane-Freespace-Detector-python/models/velloai_models/yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")

camera_feed = cv2.VideoCapture(0)

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

        frame_count += 1
        print(f"Processing frame {frame_count}")

        processed_frame = Run(model, frame)

        # detections, t = model_yolo.Inference(processed_frame)

        # print("FPS: {} sec".format(1/t))

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