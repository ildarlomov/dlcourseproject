FROM registry.gitlab.com/mcs2019/compute-base:python3.6

#docker run -it --name mcs-submit registry.gitlab.com/mcs2019/compute-base:python3.6

COPY . /opt/msc2019/public

WORKDIR /opt/msc2019/public

RUN chmod +x run.sh
CMD ./run.sh
