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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0,1,2,3,4,5,6,7', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=200, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=200, help='local train batch size')
parser.add_argument('-mn', '--modelname', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=200, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-of', '--obeserve_file', type=str, default='test_run', help='file for obeservations')
parser.add_argument('-mig', '--migration', type=int, default=1, help='enable migration')

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


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

        myClients = clients(args.num_of_clients, datasetname,
                            args.batchsize, args.epoch, sess, train, inputsx, inputsy, is_IID=args.IID)
        
        #@measure_energy(domains=[RaplCoreDomain(0)], handler=energy_csv)
        def client_local_update(*args):
            return myClients.ClientUpdate(*args)
        
        
        # params for cloud only topology
        global_energy = 0.0
        global_energy_rate = 1000.0
        myClients.base_lat = 500
        myClients.lat_factor = 10
        
        vars = tf.trainable_variables()
        global_vars = sess.run(vars)
        num_in_comm = int(max(args.num_of_clients * args.cfraction, 1))
        max_comm_lat = 0.0
        for i in tqdm(range(args.num_comm)):
            #print("communicate round {}".format(i))
            order = np.arange(args.num_of_clients)
            np.random.shuffle(order)
            clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

            sum_vars = None
            max_lat = 0.0
            for client in clients_in_comm:
                # add latency call
                curr_lat = myClients.getlat(client, len(clients_in_comm), args.migration)
                max_lat += curr_lat
                local_vars = client_local_update(client, global_vars)
                #with EnergyContext(handler=energy_csv, domains=[RaplCoreDomain(0)]) as ctx:    
                if sum_vars is None:
                    sum_vars = local_vars
                else:
                    for sum_var, local_var in zip(sum_vars, local_vars):
                       sum_var += local_var
            max_lat = max_lat / len(clients_in_comm)
            max_comm_lat = max([max_comm_lat, max_lat])

            global_vars = []
            for var in sum_vars:
                global_vars.append(var / num_in_comm)
            global_energy += global_energy_rate + random.uniform(0, global_energy_rate/2)

            if i % args.val_freq == 0:
                for variable, value in zip(vars, global_vars):
                    variable.load(value, sess)
                test_data = myClients.test_data
                test_label = myClients.test_label
                acc, cross, y_pred_run, y_true_run = sess.run([accuracy, Cross_entropy, y_pred, y_true], feed_dict={inputsx: test_data, inputsy: test_label})
                
                # data to note
                print('communication round:', i)
                print('Accuracy:', acc, 'Loss:', cross)
                print('Energy consumed:', global_energy + myClients.energy)
                print('Latency:', max_lat)
                mig_cost = myClients.getmodcost()
                print('Migration cost:', mig_cost)
                f1_val = f1_score(y_true_run, y_pred_run, average='macro')
                prec_val = precision_score(y_true_run, y_pred_run, average='macro')
                rec_val = recall_score(y_true_run, y_pred_run, average='macro')
                print(f'f1={f1_val} precision={prec_val} recall={rec_val}')
                
                # save data in dictionary
                data_note = {
                    "round": i,
                    "accuracy": float(acc),
                    "loss": [float(l) for l in cross],
                    "energy_global": global_energy,
                    "energy_client": myClients.energy,
                    "latency": max_lat,
                    "f1": f1_val,
                    "prec": prec_val,
                    "rec": rec_val,
                    "mig_cost": mig_cost
                }
                data_to_save.append(data_note)

            if i % args.save_freq == 0:
                checkpoint_name = os.path.join(args.save_path, '{}_comm'.format(args.modelname) +
                                               'IID{}_communication{}'.format(args.IID, i+1)+ '.ckpt')
                save_path = saver.save(sess, checkpoint_name)

        # save obeserved_data
        final_data = {
            "model": "fedavg",
            "val_freq": args.val_freq,
            "migration": args.migration,
            "num_clients": myClients.num_of_clients,
            "servers": 1,
            "serv_frac": 1.0,
            "client_frac": args.cfraction,
            "data": data_to_save

        }
        with open(f'./obeserve/{args.obeserve_file}', 'w') as f:
            json.dump(final_data, f)
            print('+++ written to file')
        #energy_csv.save_data()