#!/usr/bin/env python3
from lxml import etree
import random

# File paths
file_path = "/home/rchawla/catkin_ws/src/worlds/scripts/charlotteCopy.world"
output_file = '/opt/ros/noetic/share/mrs_gazebo_common_resources/worlds/grass_plane.world'

# Define the range for randomization
x_range = (-6, 6)
y_range = (0, 8)

# Function to generate random positions
def generate_random_pose():
    x_random = f"{random.uniform(*x_range):.5f}"
    y_random = f"{random.uniform(*y_range):.5f}"
    return x_random, y_random

# Number of car wheels to add
number_tyres = int(input("Enter the number of tyres in the platform: "))

# Parse the input XML file
with open(file_path, 'rb') as f:
    tree = etree.parse(f)
    root = tree.getroot()

# Detailed model template inspired by provided example
model_template = '''
<model name='disk_part{i}'>
  <link name='link'>
    <inertial>
      <pose>0 0 0.02895 0 -0 0</pose>
      <mass>0.5</mass>
      <inertia>
        <ixx>0.00321218</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>0.00321218</iyy>
        <iyz>0</iyz>
        <izz>0.00614499</izz>
      </inertia>
    </inertial>
    <collision name='collision'>
      <pose>0 0 0.02895 0 -0 0</pose>
      <geometry>
        <cylinder>
          <length>0.0579</length>
          <radius>0.15678</radius>
        </cylinder>
      </geometry>
      <max_contacts>10</max_contacts>
      <surface>
        <contact>
          <ode/>
        </contact>
        <bounce/>
        <friction>
          <torsional>
            <ode/>
          </torsional>
          <ode/>
        </friction>
      </surface>
    </collision>
    <visual name='visual'>
      <geometry>
        <mesh>
          <uri>file:///home/rchawla/gazebo_models/disk_part/meshes/disk.dae</uri>
        </mesh>
      </geometry>
      <material>
        <script>
          <uri>model://arm_part/materials/scripts</uri>
          <uri>model://arm_part/materials/textures</uri>
          <name>ArmPart/Diffuse</name>
        </script>
      </material>
    </visual>
    <self_collide>0</self_collide>
    <enable_wind>0</enable_wind>
    <kinematic>0</kinematic>
  </link>
  <pose>{x} {y} 0 0 -0 0</pose>
</model>
'''

# Find the world element to append new models
world = root.find(".//world")

# Add or update models based on user input
for i in range(1, number_tyres + 1):
    model_name = f"disk_part{i}"
    # Check if the model already exists
    existing_model = world.find(f".//model[@name='{model_name}']")
    if not existing_model:
        # Generate random positions
        x_random, y_random = generate_random_pose()
        # Create a new model with random pose
        new_model = etree.fromstring(model_template.format(i=i, x=x_random, y=y_random))
        world.append(new_model)
        print(f"Added {model_name} at pose: {x_random}, {y_random}")
    else:
        print(f"{model_name} already exists in the file.")

# Save the updated file
tree.write(output_file, encoding="UTF-8", xml_declaration=True)
print(f"File updated: {output_file}")