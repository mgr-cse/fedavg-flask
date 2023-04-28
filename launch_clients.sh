#!/bin/bash

source /home/mattie/miniconda3/bin/activate FedAvg
python client.py -id client0 --port 4000 &
python client.py -id client1 --port 4001 &
python client.py -id client2 --port 4002 &
python client.py -id client3 --port 4003 &

wait
