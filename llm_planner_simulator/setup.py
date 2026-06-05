from setuptools import find_packages, setup

package_name = 'llm_planner_simulator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'pick_and_place_simulator = llm_planner_simulator.pick_and_place_sim:main',
        ],
    },
)
