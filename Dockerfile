FROM ubuntu:18.04
RUN mkdir /lab
WORKDIR /lab
COPY . /lab
# Install system packages (python 3.5)
RUN apt-get -y update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get -qq update && apt-get -qq install --no-install-recommends -y python3.7 \ 
 python3-dev \
 python-pil \
 python-lxml \
 python-tk \
 build-essential \
 cmake \ 
 git \ 
 libgtk2.0-dev \ 
 pkg-config \ 
 libavcodec-dev \ 
 libavformat-dev \ 
 libswscale-dev \ 
 libtbb2 \
 libtbb-dev \ 
 libjpeg-dev \
 libpng-dev \
 libtiff-dev \
 libdc1394-22-dev \
 x11-apps \
 wget \
 vim \
 ffmpeg \
 unzip \
 libcanberra-gtk-module \
 && rm -rf /var/lib/apt/lists/* 

# Install core packages (TF v1.15.2)
RUN wget -q -O /tmp/get-pip.py --no-check-certificate https://bootstrap.pypa.io/get-pip.py && python3.7 /tmp/get-pip.py
RUN  pip install -U pip \
 numpy \
 matplotlib \
 notebook \
 jupyter \
 pandas \
 moviepy \
 autovizwidget

RUN pip install -r requirements.txt

# Add dataframe display widget
# RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

# Download & build OpenCV 3.4.1
# RUN wget -q -P /usr/local/src/ --no-check-certificate https://github.com/opencv/opencv/archive/3.4.1.zip
# RUN cd /usr/local/src/ \
#  && unzip 3.4.1.zip \
#  && rm 3.4.1.zip \
#  && cd /usr/local/src/opencv-3.4.1/ \
#  && mkdir build \
#  && cd /usr/local/src/opencv-3.4.1/build \ 
#  && cmake -D CMAKE_INSTALL_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local/ .. \
#  && make -j4 \
#  && make install \
#  && rm -rf /usr/local/src/opencv-3.4.1
RUN pip install opencv-python
RUN pip install imutils

# Setting up working directory 


# Minimize image size 
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)

EXPOSE 8080

ENTRYPOINT [ "python3.7" ]

CMD ["video_detection.py"]