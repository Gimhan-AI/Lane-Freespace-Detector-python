import asyncio
import random
from mavsdk import System
from mavsdk.offboard import (Attitude, OffboardError)

async def run():
    """Does Offboard control using dynamic steering and random throttle."""

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

    # Steering control parameters
    center_angle = 75
    leftmost_angle = 30
    rightmost_angle = 120
    angle = center_angle
    step = 1  # Increment/Decrement step
    moving_to = 'left'  # Start moving to the leftmost

    try:
        while True:
            throttle = random.uniform(0.4, 1.0)  # Random throttle value between 0.4 and 1.0
            print(f"-- Setting steering angle to {angle} and throttle to {throttle:.2f}")
            await drone.offboard.set_attitude(Attitude(0.0, 0.0, angle, throttle))
            await asyncio.sleep(0.1)  # Control the update frequency

            if moving_to == 'left':
                if angle > leftmost_angle:
                    angle -= step  # Move towards leftmost
                else:
                    moving_to = 'right'  # Switch to move towards rightmost

            elif moving_to == 'right':
                if angle < rightmost_angle:
                    angle += step  # Move towards rightmost
                else:
                    moving_to = 'center'  # Switch to move back to center

            elif moving_to == 'center':
                if angle > center_angle:
                    angle -= step  # Move back towards center
                elif angle < center_angle:
                    angle += step  # Adjust to reach center
                else:
                    break  # Stop after reaching center

    finally:
        print("-- Disarming")
        await drone.action.disarm()

        print("-- Stopping offboard")
        try:
            await drone.offboard.stop()
        except OffboardError as error:
            print(f"Stopping offboard mode failed with error code: {error._result.result}")

        await drone.action.kill()

if __name__ == "__main__":
    asyncio.run(run())
