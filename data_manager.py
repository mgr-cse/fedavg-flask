from flask import Flask
from flask import request
from flask import send_file
import argparse
import numpy as np
import os
import pickle

from clients import clients, user
from dataSets import DataSet

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-nc', '--num_of_clients', type=int, default=4, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

if __name__ == '__main__':
    args = parser.parse_args()
    
    dataset = DataSet('mnist', args.IID)
    dataset_size = dataset.train_data_size
    test_data = dataset.test_data
    test_label = dataset.test_label
    
    localDataSize = dataset_size // args.num_of_clients
    shard_size = localDataSize // 2
    shards_id = np.random.permutation(dataset_size // shard_size)

    clientsSet = {}
    for i in range(args.num_of_clients):
        shards_id1 = shards_id[i * 2]
        shards_id2 = shards_id[i * 2 + 1]
        data_shards1 = dataset.train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
        data_shards2 = dataset.train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
        label_shards1 = dataset.train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
        label_shards2 = dataset.train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
        someone = user(np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2)), 0)
        clientsSet[f'client{i}'] = someone
    
    test_mkdir('client-datasets')

    for k, v in clientsSet.items():
        with open(f'client-datasets/{k}.data', 'wb') as f:
            pickle.dump(v, f)
    
    with open(f'client-datasets/test.data', 'wb') as f:
        pickle.dump(test_data, f)
    
    with open(f'client-datasets/test.label', 'wb') as f:
        pickle.dump(test_label, f)
    
    print('+++ done')
    





'''

app = Flask(__name__)


@app.route('/')
def index():
    return 'Web App with Python Flask!'

    
if __name__ == '__main__':
     app.run(host='0.0.0.0', port=4000, debug=False, threaded=False, processes=1)
'''