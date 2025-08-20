#!/usr/bin/python3
import rospy
import numpy as np
import math
import random
from mrs_msgs.srv import PathSrv, PathSrvRequest
from mrs_msgs.srv import Vec1, Vec1Response
from mrs_msgs.msg import Reference
from mrs_msgs.msg import UavStatusShort
from std_msgs.msg import String
from geometry_msgs.msg import Point
from geometry_msgs.msg import QuaternionStamped
from tf.transformations import euler_from_quaternion
import cv2 # OpenCV library for computer vision
from cv_bridge import CvBridge, CvBridgeError # Converts ROS Image messages and OpenCV format
from sensor_msgs.msg import Image # ROS Message type for camera data

class MultiUAVCoordination:
    def __init__(self):
        rospy.init_node("multi_uav_coordination", anonymous = True)
        self.uav_name = rospy.get_namespace().strip('/')
        
        ## | --------------------- load parameters -------------------- |
        # Original trajectory parameters
        self.frame_id = rospy.get_param("~frame_id")
        self.center_x = float(rospy.get_param("~center/x"))
        self.center_y = float(rospy.get_param("~center/y"))
        self.center_z = float(rospy.get_param("~center/z"))
        self.dimensions_x = float(rospy.get_param("~dimensions/x"))
        self.dimensions_y = float(rospy.get_param("~dimensions/y"))
        self.trajectory_type = rospy.get_param("~trajectory_type", "random")
        
        # Disc detection parameters
        self.detection_threshold = int(rospy.get_param("~detection_threshold", 3))
        
        # Random trajectory parameters - Use full world space by default
        self.search_area_min_x = float(rospy.get_param("~search_area/min_x", -10.0))
        self.search_area_max_x = float(rospy.get_param("~search_area/max_x", 10.0))
        self.search_area_min_y = float(rospy.get_param("~search_area/min_y", -10.0))
        self.search_area_max_y = float(rospy.get_param("~search_area/max_y", 10.0))
        self.min_point_distance = float(rospy.get_param("~min_point_distance", 10.0))
        self.num_trajectory_points = int(rospy.get_param("~num_trajectory_points", 25))
        self.grid_coverage_enabled = rospy.get_param("~grid_coverage_enabled", False)

        # UAV altitude assignment for search phase
        self.altitude_map = {
            "uav1": 3.0,
            "uav2": 6.0,
            "uav3": 9.0
        }
        self.assigned_altitude = self.altitude_map.get(self.uav_name, 6.0)  # Default to 6.0 if UAV name not found
        
        # Vertical formation parameters with same offsets as your original formation
        self.vertical_formation_positions = {
            'uav1': {'z': 6.0, 'x_offset': -2, 'y_offset': -4},  # Bottom
            'uav2': {'z': 3.0, 'x_offset': 0, 'y_offset': -2},  # Middle 
            'uav3': {'z': 9.0, 'x_offset': 2, 'y_offset': -4}   # Top
        }
        
        self.is_initialized = False
        
        # Random trajectory state
        self.visited_points = []  # Store all visited (x,y) coordinates
        self.current_trajectory_active = False
        self.grid_sectors = []  # Track which grid sectors have been visited
        self.initialize_grid_sectors()
        
        # Get initial position from parameters
        self.initial_x = float(rospy.get_param("~initial_position/x", 0.0))
        self.initial_y = float(rospy.get_param("~initial_position/y", 0.0))
        
        
        # GPS and heading state variables 
        self.current_gps_x = self.initial_x          # Real-time X coordinate
        self.current_gps_y = self.initial_y          # Real-time Y coordinate  
        self.current_gps_z = self.assigned_altitude  # Real-time Z coordinate
        self.current_heading = 0.0                   # Real-time heading (radians)
        self.gps_data_received = False               # Flag to know if we have valid GPS data
        
        # Log that we're starting
        rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Node initialized at altitude {self.assigned_altitude}m')
        rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Initial position will be ({self.initial_x}, {self.initial_y}, {self.assigned_altitude})')
        
        ## | --------------------- service clients -------------------- |
        self.sc_path = rospy.ServiceProxy('~path_out', PathSrv)
        
        ## | --------------------- service servers -------------------- |
        self.ss_start = rospy.Service('~start_in', Vec1, self.callbackStart)
        
        ## | -----------------------Subscribers -------------------------|
        # Camera feed for disc detection
        self.sub_camera = rospy.Subscriber(
            "/"+self.uav_name+"/rgbd/color/image_raw",
            Image,
            self.callbackCamera
        )
        
        # UAV status for real-time GPS and heading (UPDATED)
        self.sub_uav_status = rospy.Subscriber(
            "/"+self.uav_name+"/mrs_uav_status/uav_status_short",
            UavStatusShort,
            self.callbackUAVStatus
        )
        
        # Subscribe to disc coordinate messages from other UAVs (UPDATED)
        self.sub_disc_coordinates_uav1 = rospy.Subscriber(
            "/uav1/disc_coordinates", 
            String, 
            self.callbackDiscCoordinates
        )
        self.sub_disc_coordinates_uav2 = rospy.Subscriber(
            "/uav2/disc_coordinates", 
            String, 
            self.callbackDiscCoordinates
        )
        self.sub_disc_coordinates_uav3 = rospy.Subscriber(
            "/uav3/disc_coordinates", 
            String, 
            self.callbackDiscCoordinates
        )
        self.sub_orientation = rospy.Subscriber(
            "/" + self.uav_name + "/hw_api/orientation", 
            QuaternionStamped,
            self.callbackOrientation
        )
        
        ## | -----------------------Publishers --------------------------|

        self.pub_disc_coordinates = rospy.Publisher(
            "/"+self.uav_name+"/disc_coordinates", 
            String, 
            queue_size=10
        )
        
        # Publisher for visualization
        self.pub_visualization = rospy.Publisher(
            "~visualization_out",
            Image,
            queue_size=10   
        )
        
        self.is_initialized = True
        rospy.loginfo(f'[MultiUAVCoordination and Random Trajectory Generator-{self.uav_name}]: Ready and waiting...')
        
        # ------------------ state variables ---------------------------------------
        self.disc_detected = False
        self.detection_count = 0
        self.bridge = CvBridge()
        self.target_position = None
        self.disc_detector_uav = None
        self.formation_active = False
        self.current_uav_position = None
        self.disc_detection_heading = 0.0  # Direction the detecting drone was facing when it saw the disc
        self.current_yaw = 0
        self.current_pitch = 0
        self.current_roll = 0
        self.disc_coordinates_calculated = False
        self.drone_stopped = False
        self.stop_position_x = 0.0
        self.stop_position_y = 0.0
        self.stop_position_z = 0.0
        self.stop_heading = 0.0


        self.formation_x_confirmed = 0
        self.formation_y_confirmed = 0
        self.formation_z_confirmed = 0
        self.descent_initiated = False
        self.formation_done = False

        # Disc world coordinates
        self.disc_world_x = 0.0
        self.disc_world_y = 0.0 
        self.disc_world_z = 0.0
        self.coordinates_broadcast = False  # Track if we've sent coordinates
        
        # Keep the Node Running
        rospy.spin()
    
    # -----------------------End of Constructor-------------------------------------------------------------------
    
    def initialize_grid_sectors(self):
        """Initialize grid sectors for better coverage distribution"""
        # Create a grid of sectors to ensure good coverage
        grid_size_x = 4  # Number of grid divisions in X
        grid_size_y = 4  # Number of grid divisions in Y
        
        sector_width = (self.search_area_max_x - self.search_area_min_x) / grid_size_x
        sector_height = (self.search_area_max_y - self.search_area_min_y) / grid_size_y
        
        self.grid_sectors = []
        for i in range(grid_size_x):
            for j in range(grid_size_y):
                sector = {
                    'min_x': self.search_area_min_x + i * sector_width,
                    'max_x': self.search_area_min_x + (i + 1) * sector_width,
                    'min_y': self.search_area_min_y + j * sector_height,
                    'max_y': self.search_area_min_y + (j + 1) * sector_height,
                    'visited': False,
                    'center_x': self.search_area_min_x + (i + 0.5) * sector_width,
                    'center_y': self.search_area_min_y + (j + 0.5) * sector_height
                }
                self.grid_sectors.append(sector)
        
        # Shuffle sectors for random order
        random.shuffle(self.grid_sectors)
        rospy.loginfo(f'[RandomTrajectory-{self.uav_name}]: Initialized {len(self.grid_sectors)} grid sectors')
    
    def generateRandomPoint(self):
        """Generate a random (x, y) point with better coverage distribution"""
        if self.grid_coverage_enabled:
            return self.generatePureRandomPoint()
        else:
            return self.generatePureRandomPoint()
    
    def generateGridBasedPoint(self):
        """Generate points using grid-based coverage for maximum area coverage"""
        # Find next unvisited sector
        unvisited_sectors = [sector for sector in self.grid_sectors if not sector['visited']]
        
        if not unvisited_sectors:
            # All sectors visited, reset and continue with random selection
            rospy.loginfo(f'[RandomTrajectory-{self.uav_name}]: All grid sectors visited, resetting...')
            for sector in self.grid_sectors:
                sector['visited'] = False
            random.shuffle(self.grid_sectors)
            unvisited_sectors = self.grid_sectors
        
        # Select next sector
        selected_sector = unvisited_sectors[0]
        selected_sector['visited'] = True
        
        # Generate random point within selected sector
        x = random.uniform(selected_sector['min_x'], selected_sector['max_x'])
        y = random.uniform(selected_sector['min_y'], selected_sector['max_y'])
        
        # Ensure minimum distance from previously visited points
        max_attempts = 20
        attempts = 0
        
        while attempts < max_attempts:
            is_valid = True
            for visited_x, visited_y in self.visited_points:
                distance = math.sqrt((x - visited_x)**2 + (y - visited_y)**2)
                if distance < self.min_point_distance:
                    is_valid = False
                    break
            
            if is_valid:
                break
                
            # Try another point in the same sector
            x = random.uniform(selected_sector['min_x'], selected_sector['max_x'])
            y = random.uniform(selected_sector['min_y'], selected_sector['max_y'])
            attempts += 1
        
        self.visited_points.append((x, y))
        rospy.loginfo(f'[RandomTrajectory-{self.uav_name}]: Generated point ({x:.1f}, {y:.1f}) in grid sector')
        return x, y
    
    def generatePureRandomPoint(self):
        """Generate a random (x, y) point within the search area with large spacing"""
        max_attempts = 50  # Reduced attempts since we want wide spacing
        attempts = 0
        
        while attempts < max_attempts:
            x = random.uniform(self.search_area_min_x, self.search_area_max_x)
            y = random.uniform(self.search_area_min_y, self.search_area_max_y)
            
            # Check minimum distance from visited points
            is_valid = True
            for visited_x, visited_y in self.visited_points:
                distance = math.sqrt((x - visited_x)**2 + (y - visited_y)**2)
                if distance < self.min_point_distance:
                    is_valid = False
                    break
            
            if is_valid:
                self.visited_points.append((x, y))
                rospy.loginfo(f'[RandomTrajectory-{self.uav_name}]: Generated widely spaced point ({x:.1f}, {y:.1f})')
                return x, y
            
            attempts += 1
        
        # If we can't find a point with minimum distance, generate one anyway but log it
        x = random.uniform(self.search_area_min_x, self.search_area_max_x)
        y = random.uniform(self.search_area_min_y, self.search_area_max_y)
        self.visited_points.append((x, y))
        rospy.logwarn(f'[RandomTrajectory-{self.uav_name}]: Generated point ({x:.1f}, {y:.1f}) after {max_attempts} attempts - may be closer than {self.min_point_distance}m')
        return x, y

    def planSweepPath(self, step_size):
        """Plan a sweeping pattern within assigned area bounds - vertical sweeping (Y to Y)"""
        rospy.loginfo(f'[SweepingGenerator-{self.uav_name}]: Planning vertical sweeping path with step size {step_size}')
        
        path_msg = PathSrvRequest()
        path_msg.path.header.frame_id = self.frame_id
        path_msg.path.header.stamp = rospy.Time.now()
        path_msg.path.fly_now = True
        path_msg.path.use_heading = True
        
        # Use area bounds from launch file
        area_width = self.search_area_max_x - self.search_area_min_x
        area_height = self.search_area_max_y - self.search_area_min_y
        
        # Calculate number of vertical sweep lines across the width (X direction)
        num_sweeps = max(2, int(area_width / step_size) + 1)
        sweep_spacing = area_width / (num_sweeps - 1) if num_sweeps > 1 else area_width
        
        rospy.loginfo(f'[SweepingGenerator-{self.uav_name}]: Creating {num_sweeps} vertical sweep lines across X[{self.search_area_min_x}, {self.search_area_max_x}]')
        
        # Create back-and-forth vertical sweeping pattern
        for i in range(num_sweeps):
            # Calculate X position for this sweep line
            if num_sweeps == 1:
                x = (self.search_area_min_x + self.search_area_max_x) / 2.0
            else:
                x = self.search_area_min_x + (i * sweep_spacing)
            
            # Alternate sweep direction for back-and-forth pattern
            if i % 2 == 0:
                # Even sweeps: front to back
                y_start = self.search_area_min_y
                y_end = self.search_area_max_y
            else:
                # Odd sweeps: back to front
                y_start = self.search_area_max_y
                y_end = self.search_area_min_y
            
            # Add points along this vertical sweep line
            num_points_per_line = max(3, int(area_height / step_size) + 1)
            
            for j in range(num_points_per_line):
                if num_points_per_line == 1:
                    y = (y_start + y_end) / 2.0
                else:
                    y = y_start + (j / (num_points_per_line - 1)) * (y_end - y_start)
                
                # Create waypoint
                point = Reference()
                point.position.x = x
                point.position.y = y
                point.position.z = self.center_z  # Use assigned altitude
                
                # Calculate heading toward next point (along Y direction)
                if j < num_points_per_line - 1:
                    next_y = y_start + ((j + 1) / (num_points_per_line - 1)) * (y_end - y_start)
                    point.heading = math.atan2(next_y - y, 0)  # Heading along Y direction
                else:
                    point.heading = 0.0
                
                path_msg.path.points.append(point)
        
        rospy.loginfo(f'[SweepingGenerator-{self.uav_name}]: Generated {len(path_msg.path.points)} waypoints for vertical sweeping pattern')
        return path_msg  
    
    
    def planRandomTrajectory(self, radius_factor=1.0):
        """Plan a trajectory that maintains fixed altitudes"""
        path_msg = PathSrvRequest()
        path_msg.path.header.frame_id = self.frame_id
        path_msg.path.header.stamp = rospy.Time.now()
        path_msg.path.fly_now = True
        path_msg.path.use_heading = True
        
        # First waypoint maintains current altitude - no vertical movement
        initial_point = Reference()
        initial_point.position.x = self.initial_x
        initial_point.position.y = self.initial_y
        initial_point.position.z = self.assigned_altitude  # Maintain fixed altitude
        initial_point.heading = 0.0
        path_msg.path.points.append(initial_point)
        
        # Add the initial position to visited points to avoid generating points too close
        self.visited_points.append((self.initial_x, self.initial_y))
        
        rospy.loginfo(f'[RandomTrajectory-{self.uav_name}]: First waypoint at ({self.initial_x}, {self.initial_y}, {self.assigned_altitude}) - maintaining current altitude')
        
        # Generate remaining random points - X,Y vary, Z stays fixed at assigned altitude
        for i in range(self.num_trajectory_points - 1):  # -1 because we already added the initial point
            x, y = self.generateRandomPoint()
            
            point = Reference()
            point.position.x = x  # Random X coordinate
            point.position.y = y  # Random Y coordinate
            point.position.z = self.assigned_altitude  # Fixed Z altitude
            
            # Calculate heading towards next point (or 0 for last point)
            if i < self.num_trajectory_points - 2:
                # Look ahead to next point for heading calculation
                next_x, next_y = self.generateRandomPoint()
                # Put the next point back for actual use
                self.visited_points.pop()  # Remove the look-ahead point
                heading = math.atan2(next_y - y, next_x - x)
            else:
                heading = 0.0
            
            point.heading = heading
            path_msg.path.points.append(point)
        
        rospy.loginfo(f'[RandomTrajectory-{self.uav_name}]: Generated {len(path_msg.path.points)} waypoints at fixed altitude {self.assigned_altitude}m')
        rospy.loginfo(f'[RandomTrajectory-{self.uav_name}]: X,Y coordinates vary randomly, Z stays constant')
        
        return path_msg
    
    # Vertical formation planning 
    def planVerticalFormationTrajectory(self, disc_world_x, disc_world_y, disc_world_z):
        """Plan trajectory to vertical formation position around the disc coordinates"""
        path_msg = PathSrvRequest()
        path_msg.path.header.frame_id = self.frame_id
        path_msg.path.header.stamp = rospy.Time.now()
        path_msg.path.fly_now = True
        path_msg.path.use_heading = True
        self.formation_active = True
        
        # Get vertical formation position for this UAV
        formation_info = self.vertical_formation_positions[self.uav_name]
        
        # Calculate formation position directly above the disc with original offsets
        formation_x = disc_world_x + formation_info['x_offset']
        formation_y = (disc_world_y) + formation_info['y_offset']
        formation_z = formation_info['z']  # Maintain vertical stacking
        
        # Create waypoint to the vertical formation position
        point = Reference()
        point.position.x = formation_x
        point.position.y = formation_y
        point.position.z = formation_z
        point.heading = 0.0  # Face forward or maintain current heading

        self.formation_x_confirmed = formation_x
        self.formation_y_confirmed = formation_y
        self.formation_z_confirmed = formation_z
        
        path_msg.path.points.append(point)
        
        rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Planning vertical formation at ({formation_x:.1f}, {formation_y:.1f}, {formation_z:.1f}) above disc')
        
        return path_msg
    


    
    def detectDisc(self, image):
        """Detect gray disc with shape and size filtering to avoid detecting drones"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Color detection
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 50, 200])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            contour_area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Size filtering
            if not (800 < area < 4000):
                continue
                
            # Shape filtering
            if perimeter > 0:
                circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
                aspect_ratio = float(w) / h
                
                if (circularity > 0.6 and 0.7 < aspect_ratio < 1.4):
                    rospy.loginfo(f"[RandomTrajectory-{self.uav_name}]: "
                                f"Valid disc - Area: {area}, Circularity: {circularity:.2f}")
                    return True, x, y, w, h
                else:
                    rospy.loginfo(f"[RandomTrajectory-{self.uav_name}]: "
                                f"Shape rejected - Circularity: {circularity:.2f}, "
                                f"Aspect: {aspect_ratio:.2f}")
        
        return False, 0, 0, 0, 0
    
    # Broadcast disc coordinates instead of just detection
    def broadcastDiscCoordinates(self):
        """Broadcast calculated disc world coordinates to other UAVs"""
        if not self.coordinates_broadcast and self.disc_coordinates_calculated:
            # Format: COORDINATES,uav_name,disc_x,disc_y,disc_z
            coordinates_msg = f"COORDINATES,{self.uav_name},{self.disc_world_x},{self.disc_world_y},{self.disc_world_z}"
            
            msg = String()
            msg.data = coordinates_msg
            self.pub_disc_coordinates.publish(msg)
            
            self.coordinates_broadcast = True
            
            rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Broadcasting disc coordinates: ({self.disc_world_x:.3f}, {self.disc_world_y:.3f}, {self.disc_world_z:.3f})')
    
    # Activate vertical formation mode
    def activateVerticalFormation(self, disc_world_x, disc_world_y, disc_world_z, detector_uav):
        """Activate vertical formation mode and move to formation position above disc"""
        self.formation_active = True
        self.disc_detector_uav = detector_uav
        
        rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Activating vertical formation mode. Disc coordinates: ({disc_world_x:.1f}, {disc_world_y:.1f}, {disc_world_z:.1f}) from {detector_uav}')
        
        # Plan and execute vertical formation trajectory
        formation_path = self.planVerticalFormationTrajectory(disc_world_x, disc_world_y, disc_world_z)
        
        try:
            response = self.sc_path.call(formation_path)
            if response.success:
                rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Vertical formation trajectory sent successfully')
            else:
                rospy.logerr(f'[MultiUAVCoordination-{self.uav_name}]: Formation trajectory failed: {response.message}')
        except Exception as e:
            rospy.logerr(f'[MultiUAVCoordination-{self.uav_name}]: Formation trajectory service call failed: {e}')
    
    def stopDrone(self):
        """Simply stop the drone at current position"""
        try:
            # Store the stop position
            self.stop_position_x = self.current_gps_x
            self.stop_position_y = self.current_gps_y
            self.stop_position_z = self.current_gps_z
            self.stop_heading = math.radians(self.current_yaw)
            
            # Create path message to hold position
            path_msg = PathSrvRequest()
            path_msg.path.header.frame_id = self.frame_id
            path_msg.path.header.stamp = rospy.Time.now()
            path_msg.path.fly_now = True
            path_msg.path.use_heading = True
            
            # Create waypoint at stop position
            stop_point = Reference()
            stop_point.position.x = self.stop_position_x
            stop_point.position.y = self.stop_position_y
            stop_point.position.z = self.stop_position_z
            stop_point.heading = self.stop_heading
            
            path_msg.path.points.append(stop_point)
            
            # Send the hold position command ONCE
            response = self.sc_path.call(path_msg)
            
            rospy.loginfo(f'[{self.uav_name}]: Drone stopped at ({self.stop_position_x:.2f}, {self.stop_position_y:.2f}, {self.stop_position_z:.2f})')
            
        except Exception as e:
            rospy.logerr(f'[{self.uav_name}]: Error stopping drone: {str(e)}')

    def calculateDistance(self, detected_x, detected_y, detected_w, detected_h):
        """Fixed coordinate calculation for drones with different headings"""
        
        # Camera parameters
        FOCAL_LENGTH_PIXELS = 924.27
        REAL_DISC_DIAMETER_METERS = 0.31
        OPTICAL_CENTER_X = 640.5
        OPTICAL_CENTER_Y = 360.5
        CAMERA_PITCH_OFFSET = math.radians(-178.0)  # 45° down from horizontal
        
        try:
            # 1. Calculate disc center in image coordinates
            disc_center_x = detected_x + detected_w/2.0
            disc_center_y = detected_y + detected_h/2.0
            
            # 2. Convert to normalized camera coordinates
            x_cam = (disc_center_x - OPTICAL_CENTER_X) / FOCAL_LENGTH_PIXELS
            y_cam = (disc_center_y - OPTICAL_CENTER_Y) / FOCAL_LENGTH_PIXELS
            
            # 3. Create direction vector in camera frame (Z-forward, X-right, Y-down)
            camera_dir = np.array([x_cam, y_cam, 1.0])
            camera_dir /= np.linalg.norm(camera_dir)
            
            # 4. Transform to drone body frame (account for camera mounting)
            # Camera is pitched down 45 degrees (rotation about X-axis)
            pitch_cos = math.cos(CAMERA_PITCH_OFFSET)
            pitch_sin = math.sin(CAMERA_PITCH_OFFSET)
            
            body_x = camera_dir[0]  # X unchanged
            body_y = camera_dir[1] * pitch_cos - camera_dir[2] * pitch_sin
            body_z = camera_dir[1] * pitch_sin + camera_dir[2] * pitch_cos
            
            # 5. Transform to world frame (PROPER YAW ROTATION - FIXED)
            yaw_rad = math.radians(self.current_yaw)
            
            # CORRECTED: Rotate body vector by drone yaw (rotation about Z-axis)
            world_x = body_x * math.cos(yaw_rad) - body_y * math.sin(yaw_rad)
            world_y = body_x * math.sin(yaw_rad) + body_y * math.cos(yaw_rad)
            world_z = body_z
            
            # 6. Find intersection with ground plane (Z=0)
            if abs(world_z) < 1e-6:
                rospy.logwarn("Ray parallel to ground plane")
                return False
                
            t = -self.current_gps_z / world_z
            
            # 7. Calculate world coordinates (NO ARTIFICIAL 90° ROTATION)
            self.disc_world_x = self.current_gps_x + t * world_x
            self.disc_world_y = self.current_gps_y + t * world_y
            self.disc_world_z = 0.0
            
            rospy.loginfo(f"Corrected World Coordinates: X={self.disc_world_x:.3f}, Y={self.disc_world_y:.3f}")
            return True
            
        except Exception as e:
            rospy.logerr(f"Calculation error: {str(e)}")
            return False

    def planDescent(self, formation_x, formation_y, formation_z):

        """ Plan descent trajectory points """
        path_msg = PathSrvRequest()
        path_msg.path.header.frame_id = self.frame_id
        path_msg.path.header.stamp = rospy.Time.now()
        path_msg.path.fly_now = True
        path_msg.path.use_heading = True
        self.formation_active = True
        
        # Calculate formation position directly above the disc with original offsets
        formation_x = self.formation_x_confirmed
        formation_y = self.formation_y_confirmed
        formation_z = 2.5
        
        # Create waypoint to the vertical formation position
        point = Reference()
        point.position.x = formation_x
        point.position.y = formation_y
        point.position.z = formation_z

        if(self.uav_name == "uav2"):
            point.heading = 1.48353
        elif(self.uav_name == "uav1"):
            point.heading = 2.96706
        else:
            point.heading = 0 

        path_msg.path.points.append(point)
        
        rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Planning descent at ({formation_x:.1f}, {formation_y:.1f}, {formation_z:.1f})')
        
        return path_msg

    # ------------------------------callbacks-------------------------------------------
    
    def callbackUAVStatus(self, msg):
        """Update current GPS coordinates and heading from UAV status"""
        try:
            # Extract real-time position from odometry data
            self.current_gps_x = msg.odom_x  # Current X position in meters
            self.current_gps_y = msg.odom_y  # Current Y position in meters
            self.current_gps_z = msg.odom_z  # Current Z position in meters
            
            # Extract current heading
            self.current_heading = msg.odom_hdg  # Current heading in radians
            
            # Mark that we have received valid GPS data
            self.gps_data_received = True
            
            # Optional: Log GPS updates every 2 seconds (remove this in production for less spam)
            if hasattr(self, '_last_log_time'):
                current_time = rospy.Time.now()
                if (current_time - self._last_log_time).to_sec() > 2.0:  # Log every 2 seconds
                    rospy.loginfo(f'[{self.uav_name}]: GPS Update - X:{self.current_gps_x:.2f}, Y:{self.current_gps_y:.2f}, Z:{self.current_gps_z:.2f}, Heading:{math.degrees(self.current_heading):.1f}°')
                    self._last_log_time = current_time
            else:
                self._last_log_time = rospy.Time.now()

            if(abs(self.current_gps_x - self.formation_x_confirmed) < 0.5) and (abs(self.current_gps_y - self.formation_y_confirmed) < 0.5) and (abs(self.current_gps_z - self.formation_z_confirmed) < 0.5):
                self.formation_done = True
                        
            
            if(self.formation_done and not self.descent_initiated):
                self.descent_initiated = True  # Prevent multiple executions
                formation_path = self.planDescent(self.formation_x_confirmed, self.formation_y_confirmed, self.formation_z_confirmed)
                response = self.sc_path.call(formation_path)
                rospy.loginfo(f"{self.uav_name}: Descent coordinates sent to the topic")

                

        except Exception as e:
            rospy.logerr(f'[{self.uav_name}]: Error processing UAV status: {e}')
    
    def callbackStart(self, req):
        """Start the trajectory based on trajectory_type parameter"""
        if not self.is_initialized:
            return Vec1Response(False, "not initialized")
    
        param_value = req.goal
        if param_value <= 0:
            param_value = 1.0
    
        # Only start trajectory if not in formation mode
        if not self.formation_active:
            # Choose trajectory type based on parameter
            if self.trajectory_type == "sweep":
                path_msg = self.planSweepPath(param_value)
                trajectory_name = "sweep"
            elif self.trajectory_type == "random":
                path_msg = self.planSweepPath(param_value)
                trajectory_name = "random"
            else:
                # Default to sweep if unknown type
                path_msg = self.planSweepPath(param_value)
                trajectory_name = "sweep (default)"
            
            self.current_trajectory_active = True
    
            try:
                response = self.sc_path.call(path_msg)
                if response.success:
                    rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Started {trajectory_name} trajectory')
                return Vec1Response(response.success, response.message)
            except Exception as e:
                rospy.logerr(f'[MultiUAVCoordination-{self.uav_name}]: Service call failed: {e}')
                return Vec1Response(False, "service call failed")
        else:
            return Vec1Response(True, "In formation mode, ignoring trajectory start")
    
    # andle disc coordinate messages from other UAVs
    def callbackDiscCoordinates(self, msg):
        """Handle disc coordinate messages from other UAVs"""
        try:
            parts = msg.data.split(',')
            if len(parts) >= 5 and parts[0] == "COORDINATES":
                detector_uav = parts[1]
                disc_world_x = float(parts[2])
                disc_world_y = float(parts[3])
                disc_world_z = float(parts[4])
                
                # Only respond if this UAV hasn't detected the disc itself and isn't already in formation
                if not self.disc_detected and not self.formation_active:
                    rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Received disc coordinates from {detector_uav}: ({disc_world_x:.3f}, {disc_world_y:.3f}, {disc_world_z:.3f})')
                    
                    # Activate vertical formation mode
                    self.activateVerticalFormation(disc_world_x, disc_world_y, disc_world_z, detector_uav)
                    
        except Exception as e:
            rospy.logerr(f'[MultiUAVCoordination-{self.uav_name}]: Error parsing coordinate message: {e}')
    
    # UPDATED: Camera callback with coordinate broadcasting
    def callbackCamera(self, data):
        """Process camera frames to detect discs"""
        # Skip processing if we're already in formation mode OR if coordinates already calculated
        if self.formation_active or self.disc_coordinates_calculated:
            return
            
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # Detect disc
            found, x, y, w, h = self.detectDisc(cv_image)
            
            # Draw detection visualization
            if found:
                # Visualize the detection with rectangle
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(cv_image, (x + w//2, y + h//2), 5, (0, 0, 255), -1)
                cv2.putText(cv_image, f"Disc ({x}, {y})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Count consecutive detections
                self.detection_count += 1
                
                # Confirm detection after threshold
                if self.detection_count >= self.detection_threshold and not self.disc_detected:
                    self.disc_detected = True
                    
                    rospy.loginfo(f'[{self.uav_name}]: DISC DETECTED! Stopping drone...')
                    
                    # STEP 1: Stop the drone
                    self.stopDrone()
                    self.drone_stopped = True
                    
                    # STEP 2: Calculate disc coordinates
                    rospy.loginfo(f'[{self.uav_name}]: Calculating disc world coordinates...')
                    
                    if self.calculateDistance(x, y, w, h):
                        self.disc_coordinates_calculated = True
                        
                        # STEP 3: Report the coordinates
                        rospy.loginfo(f'[{self.uav_name}]: ===== DISC COORDINATE CALCULATION COMPLETE =====')
                        rospy.loginfo(f'[{self.uav_name}]: Disc World Coordinates:')
                        rospy.loginfo(f'[{self.uav_name}]: X: {self.disc_world_x:.3f} meters')
                        rospy.loginfo(f'[{self.uav_name}]: Y: {self.disc_world_y:.3f} meters') 
                        rospy.loginfo(f'[{self.uav_name}]: Z: {self.disc_world_z:.3f} meters')
                        rospy.loginfo(f'[{self.uav_name}]: ================================================')
                        
                        # Calculate distance for verification
                        distance_to_disc = math.sqrt(
                            (self.disc_world_x - self.current_gps_x)**2 + 
                            (self.disc_world_y - self.current_gps_y)**2 + 
                            (self.disc_world_z - self.current_gps_z)**2
                        )
                        rospy.loginfo(f'[{self.uav_name}]: Distance to disc: {distance_to_disc:.3f} meters')
                        
                        # STEP 4: Broadcast coordinates to other drones
                        rospy.loginfo(f'[{self.uav_name}]: Broadcasting coordinates to other drones...')
                        self.broadcastDiscCoordinates()
                        
                        # STEP 5: Move to formation position
                        rospy.loginfo(f'[{self.uav_name}]: Moving to vertical formation position...')
                        self.activateVerticalFormation(self.disc_world_x, self.disc_world_y, self.disc_world_z, self.uav_name)
                        
                    else:
                        rospy.logwarn(f'[{self.uav_name}]: Failed to calculate disc coordinates, retrying...')
                        # Reset to try again
                        self.disc_detected = False
                        self.drone_stopped = False
                        self.detection_count = 0
            else:
                # Reset detection counter if no disc is found
                self.detection_count = 0
            
            # Add status text
            if self.formation_active:
                status = "IN VERTICAL FORMATION"
            elif self.disc_coordinates_calculated:
                status = "COORDINATES SENT - MOVING TO FORMATION"
            elif self.drone_stopped and not self.disc_coordinates_calculated:
                status = "CALCULATING DISC POSITION"
            elif self.disc_detected:
                status = "DISC DETECTED - STOPPING"
            else:
                status = "RANDOM SEARCH"
                
            cv2.putText(cv_image, f"UAV: {self.uav_name} - {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(cv_image, f"Altitude: {self.current_gps_z:.1f}m | GPS: ({self.current_gps_x:.1f}, {self.current_gps_y:.1f}) | Hdg: {math.degrees(self.current_heading):.0f}°", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Publish visualization
            viz_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.pub_visualization.publish(viz_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"[RandomTrajectory-{self.uav_name}]: {e}")
        
    def callbackOrientation(self, data):
        try:
            # Debug: Check message structure (optional)
            rospy.loginfo_once(f"Message type: {type(data)}")
            
            # Extract quaternion
            self.q_x = data.quaternion.x
            self.q_y = data.quaternion.y
            self.q_z = data.quaternion.z
            self.q_w = data.quaternion.w
            
            # Convert to Euler angles
            quat_list = [self.q_x, self.q_y, self.q_z, self.q_w]
            (roll, pitch, yaw) = euler_from_quaternion(quat_list)
            
            # Convert to degrees
            self.current_roll = math.degrees(roll)
            self.current_pitch = math.degrees(pitch)
            self.current_yaw = math.degrees(yaw)
            
            # Log occasionally to reduce spam
            if hasattr(self, '_last_orientation_log'):
                current_time = rospy.Time.now()
                if (current_time - self._last_orientation_log).to_sec() > 5.0:  # Log every 5 seconds
                    rospy.loginfo(f"[{self.uav_name}]: Roll: {self.current_roll:.2f}°, Pitch: {self.current_pitch:.2f}°, Yaw: {self.current_yaw:.2f}°")
                    self._last_orientation_log = current_time
            else:
                self._last_orientation_log = rospy.Time.now()
                
        except Exception as e:
            rospy.logerr(f"[{self.uav_name}]: Error in orientation callback: {str(e)}")
    
if __name__ == '__main__':
    try:
        node = MultiUAVCoordination()
    except rospy.ROSInterruptException:
        pass