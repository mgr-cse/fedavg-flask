from flask import Flask
from flask import request
from flask import send_file
import argparse
import numpy as np
import os
import pickle

from clients import clients, user
from Models import Models
import tensorflow as tf
from traceback import print_exc
import codecs


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-id', '--client_id', type=str, default='client0', help='id of the client process')
parser.add_argument('-p', '--port', type=int, default=4000, help='port of client process')

parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=200, help='local train batch size')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


app = Flask(__name__)


@app.route('/train', methods=['POST'])
def train():
    try:
        receive = request.json
        global_vars = receive['params']
        global_vars = pickle.loads(codecs.decode(global_vars.encode(), "base64"))
    except:
        print_exc()

    local_vars = myClients.ClientUpdate(myClients.client_id, global_vars)
    params = codecs.encode(pickle.dumps(local_vars), "base64").decode()
    return {
        "status": "success",
        "params": params
    }
          


@app.route('/')
def index():
    return 'Web App with Python Flask!'

    
if __name__ == '__main__':
    args = parser.parse_args()

    # create model
    with tf.variable_scope('inputs') as scope:
        inputsx = tf.placeholder(tf.float32, [None, 784])
        inputsy = tf.placeholder(tf.float32, [None, 10])
    myModel = Models('mnist_2nn', inputsx)

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
    
    with tf.Session(config=tf.ConfigProto(
            log_device_placement=False, \
            allow_soft_placement=True, \
            gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        
        # load user data
        with open(f'client-datasets/{args.client_id}.data', 'rb') as f:
            myUser: user = pickle.load(f)
        
        myClients = clients(args.client_id, myUser, args.batchsize, args.epoch, sess, train, inputsx, inputsy, args.IID)
    

        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=False, processes=1)