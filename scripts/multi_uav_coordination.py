#!/usr/bin/python3
import rospy
import numpy as np
import math
from mrs_msgs.srv import PathSrv, PathSrvRequest
from mrs_msgs.srv import Vec1, Vec1Response
from mrs_msgs.msg import Reference


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

        

        self.is_initialized = False # tells the code whether the setup is done or not, AM I ready to work or not
        # Log that we're starting
        rospy.loginfo(f'[MultiUAVCoordination-{self.uav_name}]: Node initialized')
        #print(f'[MultiUAVCoordination-{self.uav_name}]: Node initialized')
         ## | --------------------- service clients -------------------- |
        self.sc_path = rospy.ServiceProxy('~path_out', PathSrv)
        
        ## | --------------------- service servers -------------------- |
        self.ss_start = rospy.Service('~start_in', Vec1, self.callbackStart)


       

        self.is_initialized = True
        rospy.loginfo(f'[MultiUAVCoordination and Sweeping Generator-{self.uav_name}]: Ready and waiting...')



        # Keep the Node Running
        rospy.spin()

    # -----------------------End of COnstructor-------------------------------------------------------------------

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
       
        
    


if __name__ == '__main__': # only run this if someone directly runs this file
    try:
        node = MultiUAVCoordination()   # creates and starts the node of drone
    except rospy.ROSInterruptException:  # like interrupt when cntrl + c is pressed in terminal, this helps in no crashing of entire simulation
        pass


        

