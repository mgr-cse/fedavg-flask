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
parser.add_argument('-p', '--port', type=int, default=5000, help='port of client process')


app = Flask(__name__)


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

    
@app.route('/train', methods=['POST'])
def train():
    try:
        receive = request.json
        global_vars = receive['params']
        global_vars = pickle.loads(codecs.decode(global_vars.encode(), "base64"))
    except:
        print_exc()

    all_vars = tf.trainable_variables()
    for variable, value in zip(all_vars, global_vars):
        variable.load(value, sess)
    
    for i in tqdm(range(args.val_freq)):
        #print("communicate round {}".format(i))
        order = np.arange(args.num_of_clients)
        np.random.shuffle(order)
        clients_in_comm = list(myClients.keys())

        # call clients in parallel async
        loop = asyncio.get_event_loop()
        global_vars = loop.run_until_complete(train_clients(clients_in_comm, global_vars))


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

        # have cloud config here
        cloud_address = '127.0.0.1:5000'
        
        #@measure_energy(domains=[RaplCoreDomain(0)], handler=energy_csv)

        # app begins here
        vars = tf.trainable_variables()
        global_vars = sess.run(vars)
        num_in_comm = int(max(args.num_of_clients * args.cfraction, 1))
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=False, processes=1)
        