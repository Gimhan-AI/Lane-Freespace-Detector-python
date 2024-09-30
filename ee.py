import asyncio
import cv2
import numpy as np
import torch
from models.velloai_models import LaneFreeSpaceDetector as net
from yoloDet import YoloTRT
from mavsdk import System
from mavsdk.offboard import (Attitude, OffboardError)

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

def draw_directional_arrow(image, start_point, end_point):
    if start_point and end_point:
        # Calculate the angle between start_point and end_point
        angle = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        # Convert radians to degrees for easier interpretation and adjust by adding 90 degrees to shift from atan2 range
        angle_degrees = np.degrees(angle) + 90
        
        # Normalize the angle for display based on the rover's steering configuration
        # Map from [0, 180] (standard atan2 output range adjusted) to [30, 120] (rover's steering range)
        # Assuming 0 degrees (straight up) should now correspond to 75 degrees on your rover
        normalized_angle = np.interp(angle_degrees, [0, 180], [30, 120])
        
        # Ensure the steering angle does not exceed the rover's physical constraints
        normalized_angle = np.clip(normalized_angle, 30, 120)
        
        # Calculate the endpoint for the arrow for visualization
        arrow_length = 100  # Length of the arrow for visualization
        end_point_arrow = (
            int(start_point[0] + arrow_length * np.cos(angle)),
            int(start_point[1] - arrow_length * np.sin(angle))
        )
        
        # Draw the directional arrow on the image
        cv2.arrowedLine(image, start_point, end_point_arrow, (255, 255, 0), 3, tipLength=0.3)
        
        # Display the steering angle on the image
        cv2.putText(image, f"Steering Angle: {normalized_angle:.2f} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return image, normalized_angle


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

async def control_rover(drone, steering_angle):
    # roll_command = steering_angle / 45  # Normalize the roll angle
    await drone.offboard.set_attitude(Attitude(0.0, 0.0, steering_angle, 0.5))
    await asyncio.sleep(0.1)

async def main_loop():
    drone = System()
    await drone.connect(system_address='127.0.0.1')

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    print("-- Setting initial setpoint")
    await drone.offboard.set_attitude(Attitude(0.0, 0.0, 75.0, 0.0))
    await drone.offboard.start()

    print("-- Arming")
    await drone.action.arm()

    model = net.TwinLiteNet()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load('models/velloai_models/lanefreespacemodel.pth'))
    model.eval()
    camera_feed = cv2.VideoCapture('videos/video0.mp4')
    camera_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera_feed.isOpened():
        print("Error: Could not open camera feed.")
        return

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
                processed_frame, steering_angle = draw_directional_arrow(processed_frame, start_point, closest_fs_point)
                await control_rover(drone, steering_angle)
            cv2.imshow('Processed Camera Feed', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested exit.")
                break
    finally:
        await drone.action.disarm()
        await drone.offboard.stop()
        await drone.action.kill()
        camera_feed.release()
        cv2.destroyAllWindows()
        print("Clean-up completed.")

if __name__ == "__main__":
    asyncio.run(main_loop())
