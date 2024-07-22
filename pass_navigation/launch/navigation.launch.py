#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    param_dir = LaunchConfiguration(
        'param_dir',
        default='/home/sun/pass_ws/src/pass_navigation/config/param.yaml')

    launch_description = LaunchDescription()

    launch_description.add_action(DeclareLaunchArgument(
        'param_dir',
        default_value=param_dir,
        description='Full path of parameter file'))

    navigation_node = Node(
        package='pass_navigation',
        executable='navigation.py',
        name='navigation',
        parameters=[param_dir],
        output='screen')

    launch_description.add_action(navigation_node)
    
    return launch_description
