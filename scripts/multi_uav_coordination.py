#!/usr/bin/python3
import rospy
import numpy as np
import math
import random
from mrs_msgs.srv import PathSrv, PathSrvRequest
from mrs_msgs.srv import Vec1, Vec1Response
from mrs_msgs.msg import Reference
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Point
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
        self.hit_distance_threshold = float(rospy.get_param("~hit_distance_threshold", 1.0))
        self.hit_image_threshold = float(rospy.get_param("~hit_image_threshold", 0.6))
        
        # Random trajectory parameters - Use full world space by default
        self.search_area_min_x = float(rospy.get_param("~search_area/min_x", -20.0))
        self.search_area_max_x = float(rospy.get_param("~search_area/max_x", 20.0))
        self.search_area_min_y = float(rospy.get_param("~search_area/min_y", -20.0))
        self.search_area_max_y = float(rospy.get_param("~search_area/max_y", 20.0))
        self.min_point_distance = float(rospy.get_param("~min_point_distance", 25.0))
        self.num_trajectory_points = int(rospy.get_param("~num_trajectory_points", 12))
        self.grid_coverage_enabled = rospy.get_param("~grid_coverage_enabled", False)
        
        # MODIFIED: UAV altitude assignment with your specified altitudes
        self.altitude_map = {
            "uav1": 9.0,  # Changed from 6.0 to 9.0
            "uav2": 8.0,  # Changed from 7.0 to 8.0
            "uav3": 7.0   # Remains 7.0 (was 8.0)
        }
        # MODIFIED: Remove start_altitude concept - drones will start directly at their assigned altitude
        self.assigned_altitude = self.altitude_map.get(self.uav_name, 9.0)  # Default to 9.0 if UAV name not found
        
        # Formation parameters
        self.formation_offset = rospy.get_param("~formation_offset", 2.0)
        
        self.is_initialized = False
        
        # Random trajectory state
        self.visited_points = []  # Store all visited (x,y) coordinates
        self.current_trajectory_active = False
        self.grid_sectors = []  # Track which grid sectors have been visited
        self.initialize_grid_sectors()
        
        # MODIFIED: Get current position from Gazebo for initial waypoint
        self.initial_x = float(rospy.get_param("~initial_position/x", 0.0))  # Get from launch file or default
        self.initial_y = float(rospy.get_param("~initial_position/y", 0.0))  # Get from launch file or default
        
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
        
        # Subscribe to disc detection messages from other UAVs
        self.sub_disc_detection_uav1 = rospy.Subscriber(
            "/uav1/disc_detection_info", 
            String, 
            self.callbackDiscDetectionInfo
        )
        self.sub_disc_detection_uav2 = rospy.Subscriber(
            "/uav2/disc_detection_info", 
            String, 
            self.callbackDiscDetectionInfo
        )
        self.sub_disc_detection_uav3 = rospy.Subscriber(
            "/uav3/disc_detection_info", 
            String, 
            self.callbackDiscDetectionInfo
        )
        
        # Subscribe to UAV status to get current GPS coordinates
        self.sub_uav_status = rospy.Subscriber(
            "/"+self.uav_name+"/mrs_uav_status/uav_status_short",
            rospy.AnyMsg,  # We'll parse this manually since we don't have the exact message type
            self.callbackUAVStatus
        )
        
        ## | -----------------------Publishers --------------------------|
        # Publisher for disc detection notifications with position info
        self.pub_disc_detection_info = rospy.Publisher(
            "/"+self.uav_name+"/disc_detection_info", 
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
        
        # Current GPS coordinates (updated from UAV status)
        self.current_gps_x = self.initial_x
        self.current_gps_y = self.initial_y
        self.current_gps_z = self.assigned_altitude
        
        # Formation positions based on UAV name (diagonal formation relative to detector)
        self.formation_positions = {
            'uav1': {'z': 7.0, 'x_offset': 0.0, 'y_offset': 0.0},
            'uav2': {'z': 8.0, 'x_offset': 1.5, 'y_offset': 1.5},
            'uav3': {'z': 9.0, 'x_offset': 3.0, 'y_offset': 3.0}
        }
        
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
            return self.generateGridBasedPoint()
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
    
    def planRandomTrajectory(self, radius_factor=1.0):
        """MODIFIED: Plan a trajectory that maintains fixed altitudes - no initial ascent"""
        path_msg = PathSrvRequest()
        path_msg.path.header.frame_id = self.frame_id
        path_msg.path.header.stamp = rospy.Time.now()
        path_msg.path.fly_now = True
        path_msg.path.use_heading = True
        
        # MODIFIED: First waypoint maintains current altitude - no vertical movement
        initial_point = Reference()
        initial_point.position.x = self.initial_x
        initial_point.position.y = self.initial_y
        initial_point.position.z = self.assigned_altitude  # Maintain fixed altitude (9m, 8m, or 7m)
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
            point.position.z = self.assigned_altitude  # Fixed Z altitude (9m, 8m, or 7m)
            
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
    
    def planFormationTrajectory(self, detector_gps_x, detector_gps_y, detector_gps_z, disc_heading):
        """Plan trajectory to formation position around the detector's GPS coordinates"""
        path_msg = PathSrvRequest()
        path_msg.path.header.frame_id = self.frame_id
        path_msg.path.header.stamp = rospy.Time.now()
        path_msg.path.fly_now = True
        path_msg.path.use_heading = True
        
        # Get formation position for this UAV
        formation_info = self.formation_positions[self.uav_name]
        
        # Calculate formation offset relative to disc direction
        # Rotate the formation offsets based on the disc detection heading
        offset_x = formation_info['x_offset']
        offset_y = formation_info['y_offset']
        
        # Apply rotation to position formation behind/to the side of the detector
        # This creates a diagonal formation extending back from the disc direction
        formation_angle = disc_heading + math.pi/4  # 45 degrees off the disc direction
        
        rotated_x = offset_x * math.cos(formation_angle) - offset_y * math.sin(formation_angle)
        rotated_y = offset_x * math.sin(formation_angle) + offset_y * math.cos(formation_angle)
        
        # Calculate final formation position relative to detector's GPS
        final_x = detector_gps_x + rotated_x
        final_y = detector_gps_y + rotated_y
        final_z = formation_info['z']
        
        # Create waypoint to the formation position
        point = Reference()
        point.position.x = final_x
        point.position.y = final_y
        point.position.z = final_z
        
        # All drones face the same direction as the detector saw the disc
        point.heading = disc_heading
        
        path_msg.path.points.append(point)
        
        rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Planning formation at ({final_x:.1f}, {final_y:.1f}, {final_z:.1f}) facing {math.degrees(disc_heading):.1f}°')
        
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
    
    def broadcastDiscDetection(self, detection_x, detection_y, drone_heading):
        """Broadcast disc detection with current GPS coordinates and heading"""
        # Use current GPS position (this will be updated by UAV status)
        detection_msg = f"DETECTED,{self.uav_name},{self.current_gps_x},{self.current_gps_y},{self.current_gps_z},{drone_heading}"
        
        msg = String()
        msg.data = detection_msg
        self.pub_disc_detection_info.publish(msg)
        
        rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Broadcasting disc detection at GPS ({self.current_gps_x:.1f}, {self.current_gps_y:.1f}, {self.current_gps_z:.1f}) heading {math.degrees(drone_heading):.1f}°')
    
    def activateFormation(self, detector_uav, detector_gps_x, detector_gps_y, detector_gps_z, disc_heading):
        """Activate formation mode and move to formation position relative to detector GPS"""
        self.formation_active = True
        self.disc_detector_uav = detector_uav
        self.target_position = (detector_gps_x, detector_gps_y, detector_gps_z)
        self.disc_detection_heading = disc_heading
        
        rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Activating formation mode. Detector GPS: ({detector_gps_x:.1f}, {detector_gps_y:.1f}, {detector_gps_z:.1f}), Disc heading: {math.degrees(disc_heading):.1f}°')
        
        # Plan and execute formation trajectory
        formation_path = self.planFormationTrajectory(detector_gps_x, detector_gps_y, detector_gps_z, disc_heading)
        
        try:
            response = self.sc_path.call(formation_path)
            if response.success:
                rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Formation trajectory sent successfully')
            else:
                rospy.logerr(f'[MultiUAVCoordination-{self.uav_name}]: Formation trajectory failed: {response.message}')
        except Exception as e:
            rospy.logerr(f'[MultiUAVCoordination-{self.uav_name}]: Formation trajectory service call failed: {e}')
    
    # ------------------------------callbacks-------------------------------------------
    
    def callbackUAVStatus(self, msg):
        """Update current GPS coordinates from UAV status - simplified version"""
        # This is a simplified callback - in practice you'd parse the actual status message
        # For now, we'll approximate GPS as changing slowly from initial position
        # In real implementation, you'd extract position from mrs_uav_status/uav_status_short
        
        # For simulation purposes, we'll update GPS coordinates periodically
        # This would normally come from the actual UAV status message
        if not self.formation_active:
            # During random flight, GPS coordinates change as drone moves
            # This is a simplified approximation - real implementation would parse the message
            pass
    
    def callbackStart(self, req):
        """Start the random trajectory"""
        if not self.is_initialized:
            return Vec1Response(False, "not initialized")
    
        param_value = req.goal
        if param_value <= 0:
            param_value = 1.0
    
        # Only start trajectory if not in formation mode
        if not self.formation_active:
            path_msg = self.planRandomTrajectory(param_value)
            self.current_trajectory_active = True
    
            try:
                response = self.sc_path.call(path_msg)
                if response.success:
                    rospy.loginfo(f'[RandomTrajectory-{self.uav_name}]: Started random trajectory with {self.num_trajectory_points} points')
                return Vec1Response(response.success, response.message)
            except Exception as e:
                rospy.logerr(f'[RandomTrajectory-{self.uav_name}]: Service call failed: {e}')
                return Vec1Response(False, "service call failed")
        else:
            return Vec1Response(True, "In formation mode, ignoring trajectory start")
    
    def callbackDiscDetectionInfo(self, msg):
        """Handle disc detection messages from other UAVs with GPS coordinates and heading"""
        try:
            parts = msg.data.split(',')
            if len(parts) >= 6 and parts[0] == "DETECTED":
                detector_uav = parts[1]
                detector_gps_x = float(parts[2])
                detector_gps_y = float(parts[3])
                detector_gps_z = float(parts[4])
                disc_heading = float(parts[5])
                
                # If this UAV hasn't detected the disc itself and isn't already in formation
                if not self.disc_detected and not self.formation_active:
                    rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Received disc detection from {detector_uav} at GPS ({detector_gps_x:.1f}, {detector_gps_y:.1f}, {detector_gps_z:.1f})')
                    self.activateFormation(detector_uav, detector_gps_x, detector_gps_y, detector_gps_z, disc_heading)
                    
        except Exception as e:
            rospy.logerr(f'[MultiUAVCoordination-{self.uav_name}]: Error parsing detection message: {e}')
    
    def callbackCamera(self, data):
        """Process camera frames to detect discs"""
        # Skip processing if we're already in formation mode
        if self.formation_active:
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
                    
                    # Calculate heading towards the disc based on image position
                    image_center_x = cv_image.shape[1] / 2
                    disc_center_x = x + w/2
                    
                    # Simple heading calculation: disc position relative to image center
                    # Positive angle = disc is to the right, negative = disc is to the left
                    angle_offset = (disc_center_x - image_center_x) / image_center_x * 0.5  # Max 0.5 radians offset
                    
                    # Current drone heading (simplified - in practice get from UAV status)
                    current_heading = 0.0  # This should come from UAV status in real implementation
                    disc_heading = current_heading + angle_offset
                    
                    rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Disc detected! Heading towards disc: {math.degrees(disc_heading):.1f}°')
                    
                    # Update current GPS coordinates (simplified - should come from UAV status)
                    self.current_gps_x = self.initial_x  # In practice, get real GPS from UAV status
                    self.current_gps_y = self.initial_y
                    self.current_gps_z = self.assigned_altitude
                    
                    # Broadcast detection with GPS and heading
                    self.broadcastDiscDetection(x, y, disc_heading)
                    self.activateFormation(self.uav_name, self.current_gps_x, self.current_gps_y, self.current_gps_z, disc_heading)
            else:
                # Reset detection counter if no disc is found
                self.detection_count = 0
            
            # Add status text with altitude and formation info
            if self.formation_active:
                if self.disc_detector_uav == self.uav_name:
                    status = "FORMATION LEADER"
                else:
                    status = f"FORMATION WITH {self.disc_detector_uav}"
            else:
                status = "DISC DETECTED" if self.disc_detected else "RANDOM SEARCH"
                
            cv2.putText(cv_image, f"UAV: {self.uav_name} - {status}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(cv_image, f"Altitude: {self.assigned_altitude}m | GPS: ({self.current_gps_x:.1f}, {self.current_gps_y:.1f})", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Publish visualization
            viz_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.pub_visualization.publish(viz_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"[RandomTrajectory-{self.uav_name}]: {e}")

if __name__ == '__main__':
    try:
        node = MultiUAVCoordination()
    except rospy.ROSInterruptException:
        pass