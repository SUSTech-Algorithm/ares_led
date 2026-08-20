from glob import glob
from setuptools import setup

package_name = 'rgb_comm_protocol'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'PyYAML'],
    zip_safe=True,
    maintainer='robocon',
    maintainer_email='robocon@example.com',
    description='Shared fixed color-pair protocol for R1/R2 optical communication.',
    license='MIT',
)
