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


## TODO:
- [ ] #1
- [ ] #2

## Reference
- [carla_ros_bridge_docker](https://github.com/atinfinity/carla_ros_bridge_docker)
