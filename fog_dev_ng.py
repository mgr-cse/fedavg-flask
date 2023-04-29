import os
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from Models import Models
from clients import clients, user
import random
from sklearn.metrics import f1_score, precision_score, recall_score
import json
from pyJoules.energy_meter import measure_energy
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplCoreDomain
from pyJoules.handler.csv_handler import CSVHandler
import requests
import pickle
from traceback import print_exc
import codecs
import asyncio
from flask import Flask, request
import time
from copy import deepcopy
import threading

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0,1,2,3,4,5,6,7', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=2, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=1.0, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=200, help='local train batch size')
parser.add_argument('-mn', '--modelname', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=200, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-of', '--obeserve_file', type=str, default='test_run', help='file for obeservations')
parser.add_argument('-mig', '--migration', type=int, default=1, help='enable migration')
parser.add_argument('-fid', '--fog_id', type=str, default='fog0', help='fog device id')
parser.add_argument('-nfog', '--num_fog', type=int, default=4, help='number of fog devices')
parser.add_argument('-p', '--port', type=int, default=5000, help='port of client process')
parser.add_argument('-srf', '--server_fraction', type=float, default=1.0, help='fraction of other servers selected')

import logging
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def train_client(address: str, params):
    dump_vars = codecs.encode(pickle.dumps(params), "base64").decode()
    try:
        res = requests.post(f'http://{address}/train', json={"params": dump_vars})
        response = res.json()
        local_vars = response['params']
        local_vars = pickle.loads(codecs.decode(local_vars.encode(), "base64"))
        return local_vars
    except:
        print_exc()


async def train_clients(clients_in_comm, global_vars):
    sum_vars = None
    loop = asyncio.get_event_loop()
    local_var_futures = []
    for client in clients_in_comm:
        local_var_futures.append(loop.run_in_executor(None, train_client ,myClients[client], global_vars))

    print('train clients', len(local_var_futures))
    for f in local_var_futures:
        local_vars = await f    
        if sum_vars is None:
            sum_vars = local_vars
        else:
            for sum_var, local_var in zip(sum_vars, local_vars):
               sum_var += local_var       
    
    global_vars_new = []
    for var in sum_vars:
        global_vars_new.append(var / len(clients_in_comm))
    return global_vars_new

    
@app.route('/get_vars', methods=['POST'])
def get_vars():
    try:
        params = codecs.encode(pickle.dumps(global_vars), "base64").decode()
        return {
            "status": "success",
            "params": params
        }
    except:
        print('++++var request exception occured')
        print_exc()

@app.route('/')
def index():
    return 'Web App with Python Flask!'

@app.route('/get_f1', methods=['POST'])
def get_f1():
    for variable, value in zip(tf.trainable_variables(), global_vars):
        variable.load(value, sess)
    acc, cross, y_pred_run, y_true_run = sess.run([accuracy, Cross_entropy, y_pred, y_true], feed_dict={inputsx: test_data, inputsy: test_label})
    my_score = f1_score(y_true_run, y_pred_run, average=None)
    my_score = my_score.tolist()
    return {
        "status": "success",
        "score": my_score
    }

def get_latency(address: str):
    try:
        curr_time = time.time()
        res = requests.get(f'http://{address}/')
        if res.ok:
            lat = time.time() - curr_time
            return lat
        return 1000.0
    except:
        # return a large value
        return 1000.0

def request_f1(address: str):
    try:
        res = requests.post(f'http://{address}/get_f1', json={})
        response = res.json()
        score = response['score']
        return score
    except:
        # return a large value
        print('f1 except')
        return 'except'

def request_vars(address: str):
    try:
        res = requests.post(f'http://{address}/get_vars', json={})
        response = res.json()
        other_vars = response['params']
        return_vars = pickle.loads(codecs.decode(other_vars.encode(), "base64"))
        return return_vars
    except:
        # return a large value
        print('vars except')
        print_exc()
        return deepcopy(global_vars)

def getsim(v1, v2):
    return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def train_func():
    print('waiting for endpoints to go up')
    time.sleep(10)
    global global_vars
    all_vars = tf.trainable_variables()
    for variable, value in zip(all_vars, global_vars):
        variable.load(value, sess)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for i in tqdm(range(args.num_comm)):
        #print("communicate round {}".format(i))
        order = np.arange(args.num_of_clients)
        np.random.shuffle(order)
        clients_in_comm = list(myClients.keys())

        # call clients in parallel async
        loop = asyncio.get_event_loop()
        global_vars_updated = loop.run_until_complete(train_clients(clients_in_comm, global_vars))
        print('+++++ og vars')
        print(global_vars[0])
        print('+++++ uodated vars')
        print(global_vars_updated[0])
        global_vars = deepcopy(global_vars_updated)

        if i % args.val_freq == 0 and i != 0:
            # mixing
            for variable, value in zip(tf.trainable_variables(), global_vars_updated):
                variable.load(value, sess)
            acc, cross, y_pred_run, y_true_run = sess.run([accuracy, Cross_entropy, y_pred, y_true], feed_dict={inputsx: test_data, inputsy: test_label})
            my_score = f1_score(y_true_run, y_pred_run, average=None)
            print('accuracy:', acc)
            
            '''
            lats = []
            for fog in myFogdevs:
                lat = get_latency(myFogdevs[fog])
                print(f'latency_val: {lat}')
                if lat < lat_thresh:
                    lats.append(fog)
            f1_scores = {}
            for fog in lats:
                other_score = request_f1(myFogdevs[fog])
                if other_score == 'except':
                    other_score = my_score
                other_score = np.array(other_score)
                similarity = getsim(my_score, other_score)
                f1_scores[fog] = similarity
            
            sorted_tup = sorted(f1_scores.items(), key=lambda kv: (kv[1], kv[0]))
            num_servs = int(max(args.num_fog* args.server_fraction, 1))
            selected_servers = sorted_tup[0:num_servs]
            selected_servers = [s for s, _ in selected_servers]
            selected_servers = set(selected_servers)      
            
            serv_sum_vars = deepcopy(global_vars)
            for fog in selected_servers:
                local_vars = request_vars(myFogdevs[fog])
                for sum_var, local_var in zip(serv_sum_vars, local_vars):
                    sum_var += local_var
            global_vars_new = []
            for var in serv_sum_vars:
                global_vars_new.append(var / (len(selected_servers) + 1))

            # update the self model
            global_vars = deepcopy(global_vars_new)      
            '''

    params = codecs.encode(pickle.dumps(global_vars), "base64").decode()
    return {
        "status": "success",
        "params": params
    }
          


if __name__=='__main__':
    args = parser.parse_args()

    # GPU preparation
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    test_mkdir(args.save_path)
    test_mkdir('obeserve/')
    data_to_save = []

    if args.modelname == 'mnist_2nn' or args.modelname == 'mnist_cnn':
        datasetname = 'mnist'
        with tf.variable_scope('inputs') as scope:
            inputsx = tf.placeholder(tf.float32, [None, 784])
            inputsy = tf.placeholder(tf.float32, [None, 10])
    elif args.modelname == 'cifar10_cnn':
        datasetname = 'cifar10'
        with tf.variable_scope('inputs') as scope:
            inputsx = tf.placeholder(tf.float32, [None, 24, 24, 3])
            inputsy = tf.placeholder(tf.float32, [None, 10])

    myModel = Models(args.modelname, inputsx)

    predict_label = tf.nn.softmax(myModel.outputs)
    with tf.variable_scope('loss') as scope:
        Cross_entropy = -tf.reduce_mean(inputsy * tf.log(predict_label), axis=1)

    with tf.variable_scope('train') as scope:
        optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
        train = optimizer.minimize(Cross_entropy)

    with tf.variable_scope('validation') as scope:
        y_pred = tf.argmax(predict_label, axis=1)
        y_true = tf.argmax(inputsy, axis=1)
        correct_prediction = tf.equal(y_pred, y_true)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


    saver = tf.train.Saver(max_to_keep=3)

    # ---------------------------------------- energy --------------------------------------------- #
    #energy_csv = CSVHandler(f'obeserve/{args.obeserve_file}-energy.csv')

    # ---------------------------------------- train --------------------------------------------- #
    with tf.Session(config=tf.ConfigProto(
            log_device_placement=False, \
            allow_soft_placement=True, \
            gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        
        sess.run(tf.initialize_all_variables())

        with open('client-datasets/test.data', 'rb') as f:
            test_data = pickle.load(f)
            print('+++ test data loaded successfully')
        
        with open('client-datasets/test.label', 'rb') as f:
            test_label = pickle.load(f)
            print('+++ test label loaded successfully')


        # have client configs here
        my_id = int(args.fog_id[3:])
        myClients = {}
        base_id = args.num_of_clients*my_id
        for i in range(args.num_of_clients):
            myClients[f'client{base_id + i}'] = f'127.0.0.1:400{base_id + i}'
        print(myClients)

        # have fog config here
        lat_thresh = 10.0
        myFogdevs = {}
        for i in range(args.num_fog):
            if i == my_id:
                continue
            myFogdevs[f'fog{i}'] = f'127.0.0.1:500{i}'
        print(myFogdevs)
        
        #@measure_energy(domains=[RaplCoreDomain(0)], handler=energy_csv)

        # app begins here
        vars = tf.trainable_variables()
        global_vars = sess.run(vars)
        num_in_comm = int(max(args.num_of_clients * args.cfraction, 1))

        # start training thread
        thread1 = threading.Thread(target=train_func, args=())
        thread1.start()

        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True, processes=1)
        thread1.join()
        