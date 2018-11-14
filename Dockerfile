# RelAI docker image.

# Pull base image
FROM ubuntu:18.10

RUN apt-get update && apt-get install -y\
    software-properties-common\
    locales\
    unzip\
    python3-pip

# Install ELINA deps
RUN apt-get install -y m4 \
    libgmp-dev libgmp10\
    libmpfr-dev libmpfr-doc libmpfr6

# Set correct locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Add files.
ADD src/ /home/riai2018/
RUN cd /home/riai2018/ && unzip Archive.zip
COPY requirements.txt /home/requirements.txt

# Install ELINA
RUN cd /home/riai2018/ELINA && make && make install

# Install Gurobi
RUN cd /home/riai2018/gurobi810/linux64 && python3 setup.py install

# Install python deps
RUN pip3 install -r /home/requirements.txt

# Set environment variables.
ENV HOME /home/
ENV PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/lib:/home/riai2018/gurobi810/linux64/bin
ENV GUROBI_HOME /home/riai2018/gurobi810/linux64
ENV LD_LIBRARY_PATH /home/riai2018/gurobi810/linux64/lib:/usr/local/lib

# Define working directory.
WORKDIR /home/riai2018/analyzer

# Print sys info
RUN python3 --version
RUN pip3 --version

COPY test.sh /home/riai2018/analyzer/test.sh

# Define default command.
ENTRYPOINT ["bash", "test.sh"]
