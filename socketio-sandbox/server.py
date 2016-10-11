from flask import Flask, render_template
# documentation: http://flask.pocoo.org/docs/0.11/
# japanese: http://a2c.bitbucket.org/flask/
# repository: https://github.com/pallets/flask
from flask_socketio import SocketIO, emit
# tutorial: http://blog.miguelgrinberg.com/post/easy-websockets-with-flask-and-gevent
# documentation: https://flask-socketio.readthedocs.io/en/latest/
# repository: https://github.com/miguelgrinberg/Flask-SocketIO
import msgpack
# documentation: http://pythonhosted.org/msgpack-python/api.html
import cv2
# documentation: https://github.com/alwar/opencv3_pydocs
import threading
import time


app = Flask(__name__)
io = SocketIO(app)

@app.route('/')
def index():
    return render_template('./index.html')

@io.on('connect', namespace='/camera')
def connect():
    print("connect")

@io.on('disconnect', namespace='/camera')
def disconnect():
    print('disconnect')

@io.on('echo', namespace='/camera')
def echo(data):
    print("echo ", data)
    emit('echo', data, namespace='/camera')

@io.on('echobin', namespace='/camera')
def get(data):
    print("echobin", data)
    unpacked = msgpack.unpackb(bytearray(data["data"]))
    print("unpacked", unpacked)
    decoded = try_decode(unpacked)
    print("decoded", decoded)
    packed = msgpack.packb(decoded)
    print("packed", packed)
    emit('echobin', packed, namespace='/camera', binary=True)

def try_decode(data):
    if(type(data).__name__ == "dict"):
      _map = lambda lst: map(lambda x: try_decode(x), lst)
      _keys   = _map(data.keys())
      _values = _map(data.values())
      print(_keys, _values)
      return dict(zip(_keys, _values))
    elif(type(data).__name__ == "bytes"):
      return data.decode('utf-8')
    else:
      return data



 
class CVThread(threading.Thread):
    def __init__(self):
        self.delay = 0.3
        self.thread_stop_event = threading.Event()
        super(CVThread, self).__init__()
        self.capture = cv2.VideoCapture(0) # http://docs.opencv.org/3.1.0/d8/dfe/classcv_1_1VideoCapture.html
        if self.capture.isOpened() is False:
            raise Exception("IO Error")

    def step(self):
        while not self.thread_stop_event.isSet():
            ret, image = self.capture.read()
            if ret == False:
                return
            print(type(image)) # => <class 'numpy.ndarray'>
            data = image.tobytes("C")
            print(type(data), len(data))
            io.emit('broadcast', "hi", broadcast=True, namespace="/camera")
            time.sleep(self.delay)
 
    def run(self):
        self.step()


global thread
if __name__ == '__main__':
    print("start")
    thread = threading.Thread()

    # need visibility of the global thread object
    #Start the random number generator thread only if the thread has not been started before.
    if not thread.isAlive():
        print("Starting Thread")
        thread = CVThread()
        thread.start()

    io.run(app)
    print("end?")

