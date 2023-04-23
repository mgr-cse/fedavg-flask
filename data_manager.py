from flask import Flask
from flask import request
from flask import send_file

app = Flask(__name__)


@app.route('/')
def index():
    return 'Web App with Python Flask!'

    
if __name__ == '__main__':
     app.run(host='0.0.0.0', port=4000, debug=False, threaded=False, processes=1)