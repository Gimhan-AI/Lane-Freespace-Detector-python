import torch
import numpy as np
import cv2
import asyncio
from models.velloai_models import LaneFreeSpaceDetector as net
from yoloDet import YoloTRT
from mavsdk import System
from mavsdk.offboard import Attitude, OffboardError

# Function to process video frames
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

    # Return image in correct BGR format
    return img_rs

async def process_video(model, model_yolo):
    output_file = "model_outputs/output_processed_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    video_feed = cv2.VideoCapture('videos/video0.mp4')

    if not video_feed.isOpened():
        print("Error: Could not open video feed.")
        return

    frame_width = int(video_feed.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_feed.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_feed.get(cv2.CAP_PROP_FPS)

    print(f"Frame Width: {frame_width}, Frame Height: {frame_height}, FPS: {fps}")

    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    try:
        frame_count = 0
        while True:
            ret, frame = video_feed.read()
            if not ret:
                print("End of video stream or failed to read frame.")
                break

            frame_count += 1
            print(f"Processing frame {frame_count}")

            # Process the frame using the first model
            processed_frame = Run(model, frame)

            # Ensure processed frame size is correct
            processed_frame = cv2.resize(processed_frame, (frame_width, frame_height))

            # Run the processed frame through the second model
            detections, t = model_yolo.Inference(processed_frame)

            print("FPS: {} sec".format(1/t))

            cv2.imshow('Processed Video Feed', processed_frame)

            # Write the processed frame to the output file
            out.write(processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested exit.")
                break

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")

    finally:
        # Release the video capture and writer objects
        video_feed.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Processed video saved as {output_file}")

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
    # Initialize models
    model = net.TwinLiteNet()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load('models/velloai_models/lanefreespacemodel.pth'))
    model.eval()

    model_yolo = YoloTRT(library="/home/velloai/Desktop/LaneFreeSpaceDetector/models/velloai_models/yolov5/build/libmyplugins.so", engine="/home/velloai/Desktop/LaneFreeSpaceDetector/models/velloai_models/yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")

    # Run video processing and drone control concurrently
    await asyncio.gather(
        process_video(model, model_yolo),
        control_drone()
    )

if __name__ == "__main__":
    # Run the asyncio loop
    asyncio.run(main())