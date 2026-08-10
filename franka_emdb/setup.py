import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'franka_emdb'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # (   os.path.join('share', package_name, 'config'), glob('config/*.[png]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ines',
    maintainer_email='ruizplovin@isir.upmc.fr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 'test_node = franka_emdb.test:main',
            'franka_server = franka_emdb.franka_emdb_server:main',
        ],
    },
)
