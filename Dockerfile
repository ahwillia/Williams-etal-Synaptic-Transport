FROM andrewosh/binder-base

MAINTAINER Alex Williams <alex.h.willia@gmail.com>

RUN git clone https://github.com/jakevdp/JSAnimation.git
RUN python JSAnimation/setup.py install

ENV PYTHONPATH $PYTHONPATH:$HOME/JSAnimation/
