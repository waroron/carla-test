FROM nvidia/cudagl:11.4.2-devel-ubuntu18.04

ARG CARLA_VERSION=0.9.11
ARG BOOST_VERSION=1.80.0
ARG GID=1000
ARG UID=1000
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# add new sudo user
ENV USERNAME melodic
ENV HOME /home/$USERNAME
RUN groupadd -f -g ${GID} ${USERNAME}
RUN useradd -m $USERNAME -u ${UID} -g ${GID} && \
    echo "$USERNAME:$USERNAME" | chpasswd && \
    usermod --shell /bin/bash $USERNAME && \
    usermod -aG sudo $USERNAME && \
    mkdir /etc/sudoers.d && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    less \
    emacs \
    tmux \
    bash-completion \
    command-not-found \
    software-properties-common \
    xsel \
    xdg-user-dirs \
    python-pip \
    python-protobuf \
    python-pexpect \
    pcl-tools \
    libomp5 \
    vulkan-utils \
    wget \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Vulkan
RUN mkdir -p /etc/vulkan/icd.d && \
    echo '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libGLX_nvidia.so.0", "api_version" : "1.1.99" } }' > /etc/vulkan/icd.d/nvidia_icd.json
RUN echo > /etc/ld.so.preload

# ROS Melodic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# RUN apt install curl
# RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-desktop-full \
    python-rosdep \
    python-rosinstall \
    python-rosinstall-generator \
    python-wstool \
    build-essential \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# CARLA Simulator
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 1AF1527DE64CB8D9
RUN add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla $(lsb_release -sc) main"
RUN apt-get update && apt-get install -y --no-install-recommends \
    carla-simulator=${CARLA_VERSION} \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# addtional ROS package
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-derived-object-msgs \
    ros-melodic-ackermann-msgs \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# addtional python package
RUN pip install \
    transforms3d==0.3.1 \
    networkx==2.2 \
    pygame==1.9.6

RUN rosdep init

# for SC-LIO-SAM
RUN apt-get update
RUN add-apt-repository ppa:borglab/gtsam-release-4.0 && \
    apt install -y libgtsam-dev libgtsam-unstable-dev

RUN apt-get install -y ros-melodic-navigation ros-melodic-robot-localization ros-melodic-robot-state-publisher

# for VDBFusion
# Install OpenVDB
RUN apt-get update && apt-get install --no-install-recommends -y \
    libblosc-dev \
    libboost-iostreams-dev \
    libboost-system-dev \
    libboost-system-dev \
    libeigen3-dev

# install newer CMake manually
RUN cd /home/$USERNAME && \
    wget https://github.com/Kitware/CMake/releases/download/v3.24.0/cmake-3.24.0.tar.gz && \
    tar zxvf cmake-3.24.0.tar.gz && \
    cd cmake-3.24.0/ && \
    ./bootstrap && \
    make -j12 && make install -j8
RUN echo 'export PATH=$HOME/cmake-3.24.0/bin/:$PATH' >> ~/.bashrc && \
    . ~/.bashrc

# Install Boost
# https://www.boost.org/doc/libs/1_80_0/more/getting_started/unix-variants.html
RUN cd /tmp && \
    BOOST_VERSION_MOD=$(echo $BOOST_VERSION | tr . _) && \
    wget https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_MOD}.tar.bz2 && \
    tar --bzip2 -xf boost_${BOOST_VERSION_MOD}.tar.bz2 && \
    cd boost_${BOOST_VERSION_MOD} && \
    ./bootstrap.sh --prefix=/usr/local && \
    ./b2 install && \
    rm -rf /tmp/*

USER $USERNAME
WORKDIR /home/$USERNAME
RUN rosdep update

SHELL ["/bin/bash", "-c"]

# RUN sudo sh -c 'echo deb https://apt.repos.intel.com/tbb all main > /etc/apt/sources.list.d/intel-tbb.list' && \
#     sudo apt update && \ 
#     sudo apt install -y libtbb-dev intel-tbb

# build CARLA ROS bridge
RUN mkdir -p ~/catkin_ws/src && \
    source /opt/ros/melodic/setup.bash && \
    catkin_init_workspace ~/catkin_ws/src && \
    cd ~/catkin_ws/src && \
    git clone --recursive https://github.com/carla-simulator/ros-bridge.git -b ${CARLA_VERSION} && \
    cd ~/catkin_ws && \
    catkin_make -DCMAKE_BUILD_TYPE=Release && \
    source ~/catkin_ws/devel/setup.bash

# build CARLA ScenarioRunner
RUN cd /home/$USERNAME && \
    git clone https://github.com/carla-simulator/scenario_runner.git -b v${CARLA_VERSION} && \
    sudo pip install -r scenario_runner/requirements.txt

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc && \
    echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc && \
    echo "export SCENARIO_RUNNER_PATH=/home/$USERNAME/scenario_runner" >> ~/.bashrc

# install tbb
RUN cd /home/$USERNAME && \
    git clone https://github.com/wjakob/tbb.git  && \
    cd tbb && \
    cd build && \
    cmake .. && \
    sudo make -j$(nproc) all install

# Install
RUN cd /home/$USERNAME && \
    git clone --depth 1 https://github.com/nachovizzo/openvdb.git -b nacho/vdbfusion && \
    cd openvdb && \
    mkdir build && cd build && \
    cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DUSE_ZLIB=OFF .. && \
    sudo make -j$(nproc) all install

# Install VDBfusion
RUN cd /home/$USERNAME && \
    git clone --depth 1 https://github.com/PRBonn/vdbfusion.git && \
    cd vdbfusion && \
    mkdir build && cd build && \
    cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DUSE_ZLIB=OFF .. && \
    sudo make -j$(nproc) all install

RUN sudo apt install -y python-catkin-tools ros-melodic-geometry2 ros-melodic-slam-gmapping
