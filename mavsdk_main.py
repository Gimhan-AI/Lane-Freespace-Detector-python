#!/usr/bin/env PYTHONUNBUFFERED=1 python3

import asyncio
import signal
from mavsdk import System
from mavsdk.offboard import (Attitude, OffboardError)
from mavsdk.tune import (SongElement, TuneDescription, TuneError)
import subprocess
import time
from multiprocessing import Process, Manager
import traceback


rover = None
current_waypoint = 0
lidar_obstacle = False

async def run_LiDar(stop_lidar):
    global rover, lidar_obstacle

    threshold_dis = 800
    controller_port = "/dev/ttyUSB0"
    controller_baudrate = 115200
    OD_sent_flag = False
    distance_min = 100000
    
    rover = System()

    # Initialize the command and process for the lidar
    command_lidar = ['./ultra_simple --channel --serial /dev/ttyUSB0 1000000']
    directory_path_lidar = '/home/vegaai/rukshan/Camera_LiDar_fusion/rplidar_sdk/output/Linux/Release'
    process = subprocess.Popen(command_lidar, cwd=directory_path_lidar, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    await connect2rover()
    
    print_mission_progress_task = asyncio.ensure_future(
        print_mission_progress(rover))

    running_tasks = [print_mission_progress_task]
    

    # Wait for lidar to start
    output_line = process.stdout.readline().decode('utf-8')
    while output_line != 'Set_Start.\n':
        output_line = process.stdout.readline().decode('utf-8')

    try:
        while True:
            output_line = process.stdout.readline().decode('utf-8')

            if output_line == 'Set_Start.\n':
                if distance_min > threshold_dis and OD_sent_flag:
                    print("Armed")
                    OD_sent_flag = False  # Reset flag when rover is armed
                    try:
                        #get current waypoint if in mission mode
                        await rover.action.arm()
                        if current_waypoint !=0:
                            rover.mission.set_current_mission_item(current_waypoint)
                            rover.mission.start_mission()
                        lidar_obstacle = False
                            
                    except Exception as e:
                        print(f"Couldn't Arm: {e}")
                distance_min = 100000
            else:
                try:
                    measurements = output_line.split()
                    distance = float(measurements[1])
                    if distance < distance_min:
                        distance_min = distance

                    if distance < threshold_dis:
                        if not OD_sent_flag:
                            print("Disarmed")
                            OD_sent_flag = True  # Set flag to prevent multiple calls
                            try:
                                # if mission was ongoing set waypoint and start mission 
                                lidar_obstacle = True
                                await rover.action.disarm()
                            except Exception as e:
                                print(f"Couldn't Disarm: {e}")
                            print(output_line)

                except Exception as e:
                    pass
                    #print(f"Error processing lidar data: {e}")

            if stop_lidar.value:
                print("Stopping lidar process...")
                break

    except Exception as e:
        print(e)
        print("Run_LiDar cancelled")
    finally:
        process.kill()  # Make sure to terminate the lidar process
        print("Lidar process terminated.")
        

async def connect2rover():
    """ Connect to the rover """
    await rover.connect(system_address='127.0.0.1')

    print("Waiting for drone to connect...")
    async for state in rover.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break


async def print_mission_progress(drone):
    global current_waypoint
    async for mission_progress in drone.mission.mission_progress():
        if not(lidar_obstacle):
            current_waypoint = mission_progress.current
    
        #print(f"Mission progress: "
        #      f"{mission_progress.current}/"
        #      f"{mission_progress.total}")



async def shutdown(stop_lidar):
    """ Gracefully shutdown the lidar process """
    print("Shutting down...")
    stop_lidar.value = True  # Signal to stop the lidar task

async def main():
    with Manager() as manager:
        stop_lidar = manager.Value('b', False)
        lidar_process = Process(target=asyncio.run, args=(run_LiDar(stop_lidar),))
        lidar_process.start()
        print("Started Lidar")

        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(stop_lidar)))

        try:
            while not stop_lidar.value:
                await asyncio.sleep(0.1)

        except Exception as e:
            traceback.print_exc()
            await shutdown(stop_lidar)

        finally:
            lidar_process.join()  # Wait for the lidar process to finish

if __name__ == '__main__':
    asyncio.run(main())

