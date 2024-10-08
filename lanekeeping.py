import torch
import numpy as np
import cv2
import asyncio
from models.velloai_models import LaneFreeSpaceDetector as net
from yoloDet import YoloTRT
from mavsdk import System
from mavsdk.offboard import Attitude, OffboardError

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

    return img_rs

async def process_video(model, model_yolo):
    camera_feed = cv2.VideoCapture(0)

    camera_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera_feed.isOpened():
        print("Error: Could not open camera feed.")
        return

    try:
        frame_count = 0
        while True:
            ret, frame = camera_feed.read()
            if not ret:
                print("End of video stream or failed to read frame.")
                break

            frame_count += 1

            processed_frame = Run(model, frame)

            processed_frame = cv2.resize(processed_frame, (frame_width, frame_height))

            detections, t = model_yolo.Inference(processed_frame)

            print("FPS: {} sec".format(1/t))

            cv2.imshow('Processed Camera Feed', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested exit.")
                break

    except KeyboardInterrupt:
        print("\nProcessing stopped by user.")

    finally:
        camera_feed.release()
        cv2.destroyAllWindows()

async def control_drone():
    """ Does Offboard control using attitude commands. """

    drone = System()
    await drone.connect(system_address='127.0.0.1')

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("-- Setting initial setpoint")
    await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.0))
    await drone.offboard.start()

    print("-- Arming")
    await drone.action.arm()

    print("-- Go up at 70% thrust")
    await drone.offboard.set_attitude(Attitude(0.0, 0.0, -45, 0.7))
    await asyncio.sleep(3)

    print("-- Hover at 60% thrust")
    await drone.offboard.set_attitude(Attitude(0.0, 0.0, 10, 0.5))
    await asyncio.sleep(3)

    await drone.action.disarm()

    print("-- Stopping offboard")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed with error code: \
              {error._result.result}")

    await drone.action.kill()

async def main():
    model = net.TwinLiteNet()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load('models/velloai_models/lanefreespacemodel.pth'))
    model.eval()

    model_yolo = YoloTRT(library="/home/vegaai/Lane-Freespace-Detector-python/models/velloai_models/yolov5/build/libmyplugins.so", engine="/home/vegaai/Lane-Freespace-Detector-python/models/velloai_models/yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")

    await asyncio.gather(
        process_video(model, model_yolo),
        control_drone()
    )

if __name__ == "__main__":
    asyncio.run(main())