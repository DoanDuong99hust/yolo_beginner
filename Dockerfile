FROM ubuntu@sha256:0f744430d9643a0ec647a4addcac14b1fbb11424be434165c15e2cc7269f70f8
RUN mkdir /lab
WORKDIR /lab
COPY . /lab
# Install system packages (python 3.5)
RUN apt-get -y update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN apt-get update -y
#RUN apt-get -qq install --no-install-recommends -y python3
#RUN apt-get -qq update && apt-get -qq install --no-install-recommends -y wget vim 
#RUN apt-get install build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev libffi-dev -y
#RUN wget --no-check-certificate https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tar.xz
#RUN tar xf Python-3.7.0.tar.xz
#RUN cd Python-3.7.0 && ./configure && make -j 4 && make altinstall
#RUN cd .. && rm Python-3.7.0.tar.xz
#RUN sudo apt-get --purge remove build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev libffi-dev -y
#RUN sudo apt-get autoremove -y
#RUN sudo apt-get clean
 RUN apt-get -qq update && apt-get -qq install --no-install-recommends -y python3 \ 
 python3-dev \
# python-pil \
# python-lxml \
# python-tk \
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
# libdc1394-22-dev \
 x11-apps \
 wget \
 vim \
 ffmpeg \
 unzip \
 libcanberra-gtk-module \
 && rm -rf /var/lib/apt/lists/* 

# Install core packages (TF v1.15.2)
 RUN wget -q -O /tmp/get-pip.py --no-check-certificate https://bootstrap.pypa.io/get-pip.py && python3 /tmp/get-pip.py
#RUN apt-get -qq install --no-install-recommends -y python3 \
#libqtgui4 \
#libqt4-test

RUN pip install -U pip \
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

ENTRYPOINT [ "python3" ]

CMD ["video_detection.py"]
