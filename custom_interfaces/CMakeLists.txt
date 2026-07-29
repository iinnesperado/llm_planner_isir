cmake_minimum_required(VERSION 3.5...3.20)
project(custom_interfaces)
if(POLICY CMP0148) 
  cmake_policy(SET CMP0148 OLD) 
endif()

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  ##### APP SERVICES #####
  "srv/app/BaseAppRequest.srv"
  "srv/app/ExecuteRequest.srv"
  "srv/app/ExploreRequest.srv"
  "srv/app/GraspRequest.srv"
  "srv/app/GraspRequest2.srv"

  DEPENDENCIES geometry_msgs sensor_msgs std_msgs
 )
 
ament_export_dependencies(rosidl_default_runtime)
ament_export_dependencies(geometry_msgs)
ament_export_dependencies(sensor_msgs)
ament_export_dependencies(std_msgs)
ament_package()