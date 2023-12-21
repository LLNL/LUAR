FROM python:3.8.15-slim
LABEL AUTHOR="Rafael A. Rivera Soto <riverasoto1@llnl.gov>"

WORKDIR "/root"
RUN  mkdir LUAR
COPY . LUAR/
WORKDIR "/root/LUAR"
RUN pip3 install -U pip
RUN pip3 install -r requirements.txt
CMD ["/bin/bash"]