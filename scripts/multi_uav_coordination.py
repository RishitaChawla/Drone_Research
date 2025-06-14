#!/usr/bin/python3
import rospy
import numpy as np
import math
from mrs_msgs.srv import PathSrv, PathSrvRequest
from mrs_msgs.srv import Vec1, Vec1Response
from mrs_msgs.msg import Reference
from std_msgs.msg import Bool

import cv2 # OpenCV library for computer vision
from cv_bridge import CvBridge, CvBridgeError # Converts ROS Image messages and OpenCV format
from sensor_msgs.msg import Image # ROS Message type for camera data


class MultiUAVCoordination:
    def __init__(self):


        rospy.init_node("multi_uav_coordination", anonymous = True)



        self.uav_name = rospy.get_namespace().strip('/')
        #self.uav_ids = ["uav1", "uav2", "uav3"]

        ## | --------------------- load parameters -------------------- |
        # Original trajectory parameters
        self.frame_id = rospy.get_param("~frame_id")
        self.center_x = rospy.get_param("~center/x")
        self.center_y = rospy.get_param("~center/y")
        self.center_z = rospy.get_param("~center/z")
        self.dimensions_x = rospy.get_param("~dimensions/x")
        self.dimensions_y = rospy.get_param("~dimensions/y")
        self.trajectory_type = rospy.get_param("~trajectory_type", "sweep")


        
        # Disc detection parameters
        self.detection_threshold = rospy.get_param("~detection_threshold", 3)  # Number of consecutive detections needed
        self.hit_distance_threshold = rospy.get_param("~hit_distance_threshold", 1.0)
        self.hit_image_threshold = rospy.get_param("~hit_image_threshold", 0.6)

        

        self.is_initialized = False # tells the code whether the setup is done or not, AM I ready to work or not
        
        # Log that we're starting
        rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Node initialized')
        
         ## | --------------------- service clients -------------------- |
        self.sc_path = rospy.ServiceProxy('~path_out', PathSrv) # Request to Server
        
        ## | --------------------- service servers -------------------- |
        self.ss_start = rospy.Service('~start_in', Vec1, self.callbackStart) # Trigger to start the circular trajectory

        ## | -----------------------Subscribers -------------------------|

        # Camera feed for disc detection
        self.sub_camera = rospy.Subscriber(
            "/"+self.uav_name+"/rgbd/color/image_raw",    # Topic name
            Image,                                        # Message type
            self.callbackCamera                          # Function to call
        )


        # Publisher for disc detection notifications
        self.pub_disc_detection = rospy.Publisher(
            "/"+self.uav_name+"/disc_detection", 
            Bool, 
            queue_size=10
        )       

        # Publisher for visualization
        self.pub_visualization = rospy.Publisher(
            "~visualization_out",
            Image,
            queue_size=10   
        )       

       

        self.is_initialized = True
        rospy.loginfo(f'[MultiUAVCoordination and Sweeping Generator-{self.uav_name}]: Ready and waiting...')

        # ------------------ state variables ---------------------------------------
        self.disc_detected = False        # Have we confirmed seeing a disc?
        self.detection_count = 0          # How many times have we seen it consecutively?
        self.bridge = CvBridge()          # Tool to convert ROS images to OpenCV
        self.target_position = None       # Where was the disc when we found it?
        self.disc_detector_uav = None  # Which UAV detected the disc

        # Keep the Node Running
        rospy.spin()

    # -----------------------End of Constructor-------------------------------------------------------------------

    def planCircleTrajectory(self,radius_factor = 1.0):
        path_msg = PathSrvRequest()

        path_msg.path.header.frame_id = self.frame_id
        path_msg.path.header.stamp = rospy.Time.now()
        path_msg.path.fly_now = True
        path_msg.path.use_heading = True


        radius = self.dimensions_x / 2.0 * radius_factor
        num_points = 30

        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            point = Reference()
            point.position.x = self.center_x + radius * math.cos(angle)
            point.position.y = self.center_y + radius * math.sin(angle)
            point.position.z = self.center_z
            point.heading = angle + math.pi / 2
            path_msg.path.points.append(point)
        
        return path_msg
    

    def detectDisc(self, image):
        """
        Detect gray disc with shape and size filtering to avoid detecting drones
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Color detection
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 50, 200])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Use underscore to indicate "I don't need this variable"
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            contour_area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Size filtering (adjust these values based on your disc size)
            if not (800 < area < 4000):
                continue
                
            # Shape filtering
            if perimeter > 0:
                circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
                aspect_ratio = float(w) / h
                
                # Disc should be circular and round
                if (circularity > 0.6 and 0.7 < aspect_ratio < 1.4):
                    rospy.loginfo(f"[SweepingGenerator-{self.uav_name}]: "
                                f"Valid disc - Area: {area}, Circularity: {circularity:.2f}")
                    return True, x, y, w, h
                else:
                    rospy.loginfo(f"[SweepingGenerator-{self.uav_name}]: "
                                f"Shape rejected - Circularity: {circularity:.2f}, "
                                f"Aspect: {aspect_ratio:.2f}")
        
        return False, 0, 0, 0, 0
    def broadcastDiscDetection(self):
        """Broadcast disc detection to other UAVs"""
        msg = Bool()
        msg.data = True
        self.pub_disc_detection.publish(msg)
        rospy.loginfo(f'[SweepingGenerator-{self.uav_name}]: Broadcasting disc detection!')


    # ------------------------------callbacks-------------------------------------------

    def callbackStart(self, req): # from start_in trigger service created by user
        if not self.is_initialized:
            return Vec1Response(False, "not initialized")
    
        param_value = req.goal
        if param_value <= 0:
            param_value = 1.0
    
        path_msg = self.planCircleTrajectory(param_value)
    
        try:
            response = self.sc_path.call(path_msg)
            return Vec1Response(response.success, response.message)
        except Exception as e:
            return Vec1Response(False, "service call failed")

    def callbackCamera(self, data):
        """Process camera frames to detect discs"""
        # Skip processing if we've already detected a disc
        # or if we're responding to another UAV's detection
        if self.disc_detected or self.disc_detector_uav is not None:
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
                    self.disc_detector_uav = self.uav_name  # Mark self as the detector
                    rospy.loginfo(f'[SweepingGenerator-{self.uav_name}]: Disc detected!')
                    self.broadcastDiscDetection()
            else:
                # Reset detection counter if no disc is found
                self.detection_count = 0
            
            # Add status text
            status = "DISC DETECTED" if self.disc_detected else "SEARCHING"
            if self.disc_detector_uav and self.disc_detector_uav != self.uav_name:
                status = f"FORMATION WITH {self.disc_detector_uav}"
                
            cv2.putText(cv_image, f"UAV: {self.uav_name} - {status}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Publish visualization
            viz_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.pub_visualization.publish(viz_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"[SweepingGenerator-{self.uav_name}]: {e}")


        
if __name__ == '__main__': # only run this if someone directly runs this file
    try:
        node = MultiUAVCoordination()   # creates and starts the node of drone
    except rospy.ROSInterruptException:  # like interrupt when cntrl + c is pressed in terminal, this helps in no crashing of entire simulation
        pass


        
