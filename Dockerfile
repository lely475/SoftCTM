FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN apt-get -y update \
  && apt-get install -y software-properties-common \
  && apt-get -y update \
  && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y update
RUN apt-get -y install python3.9
RUN apt-get -y install python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN mkdir -p /opt/app /input /output \
  && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"
ENV PYTHONPATH "${PYTHONPATH}:/opt/app/"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

COPY ./ /opt/app/

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
