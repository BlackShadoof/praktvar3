FROM ubuntu:16.04
ENV DISPLAY=192.168.1.67:0.0
RUN apt-get update
RUN apt-get install python3 -y
RUN apt-get update
RUN apt-get install libopencv-dev -y
RUN apt-get install python3-pip -y
RUN apt-get install python3-tk -y
RUN pip3 install --upgrade pip

RUN pip3 install matplotlib
RUN pip3 install opencv-python
RUN pip3 install opencv-contrib-python

COPY . /app
WORKDIR /app
CMD ["python3", "main.py"]