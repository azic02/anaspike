FROM nest/nest-simulator:3.8

RUN apt-get update && apt-get install -y \
    ffmpeg

RUN echo "source /opt/nest/bin/nest_vars.sh" >> /root/.bashrc

WORKDIR /opt/data

ENTRYPOINT ["/bin/bash"]

