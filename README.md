# carla-test

## Requirements
- Docker
- NVIDIA docker

## setup

```
docker build -t carla:0.9.11 --build-arg GID=$(id -g) --build-arg UID=$(id -u) -f docker/Dockerfile .

sh docker/start.sh
```


## Usage
### Carla ROS Bridge Demo
1. Terminal 1(launch carla simulator)
```
/opt/carla-simulator/CarlaUE4.sh
```

2. Terminal 2(set the configuration)
```
cd /opt/carla-simulator/PythonAPI
python util/config.py -m Town03 --fps 10
```

3. Terminal 3(launch carla ros bridge)
```
roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch vehicle_filter:='vehicle.tesla.model3'
```



## TODO:

- [ ] #1
- [ ] #2

## Reference
- [carla_ros_bridge_docker](https://github.com/atinfinity/carla_ros_bridge_docker)
