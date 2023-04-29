#!/bin/bash

source /home/mattie/miniconda3/bin/activate FedAvg
python fog_dev_ng.py -fid fog0 --port 5000 &
python fog_dev_ng.py -fid fog1 --port 5001 &
python fog_dev_ng.py -fid fog2 --port 5002 &
python fog_dev_ng.py -fid fog3 --port 5003 &

wait