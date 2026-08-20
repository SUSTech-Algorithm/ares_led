from glob import glob
from pathlib import Path
from setuptools import setup

package_name = 'rgb_camera_receiver'


def camera_config_files():
    rows = []
    root = Path('config/cameras')
    for directory in sorted(path for path in root.rglob('*') if path.is_dir()):
        files = [str(path) for path in sorted(directory.iterdir()) if path.is_file()]
        if files:
            rows.append(('share/' + package_name + '/' + str(directory), files))
    return rows


setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ] + camera_config_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robocon',
    maintainer_email='robocon@example.com',
    description='Protocol-independent five-color LED strip detector for R2.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'rgb_camera_receiver = rgb_camera_receiver.node:main',
            'evaluate_led_dataset = rgb_camera_receiver.evaluate:main',
            'calibrate_led_colors = rgb_camera_receiver.calibrate:main',
        ],
    },
)
