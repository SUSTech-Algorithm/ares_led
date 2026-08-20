from glob import glob
from setuptools import setup

package_name = 'rgb_led_sender'
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robocon',
    maintainer_email='robocon@example.com',
    description='Encode command IDs as two WLED-controlled LED strip colors.',
    license='MIT',
    entry_points={'console_scripts': ['rgb_led_sender = rgb_led_sender.node:main']},
)
