from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    sender_share = Path(get_package_share_directory('rgb_led_sender'))
    return LaunchDescription([
        Node(
            package='rgb_led_sender',
            executable='rgb_led_sender',
            name='rgb_led_sender',
            output='screen',
            parameters=[str(sender_share / 'config/sender.yaml')],
        ),
    ])
