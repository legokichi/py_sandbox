from typing import no_type_check, no_type_check_decorator
from typing import get_type_hints, cast
from typing import TypeVar, NewType, Type, Generic#, ClassVar

from typing import Container, Hashable, Iterable, Sized, Callable, Awaitable, AsyncIterable
from typing import Iterator, Sequence, Set, Mapping, MappingView, AsyncIterator#, Coroutine
from typing import Generator, MutableSequence, ByteString, MutableSet, MutableMapping, ItemsView, KeysView, ValuesView

from typing import SupportsInt, SupportsFloat, SupportsAbs, SupportsRound
from typing import Union, Optional, AbstractSet, Reversible
from typing import Any, re, io, AnyStr, Tuple, NamedTuple, List, Dict, DefaultDict 
 

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
from PIL import Image
import numpy as np

import threading
import time
import io



app = Flask(__name__) # type: Flask
socketio = SocketIO(app) # type: SocketIO

@app.route('/')
def index()-> str:
    return render_template('./index.html')

@socketio.on('connect', namespace='/camera')
def connect():
    print("connect")

@socketio.on('disconnect', namespace='/camera')
def disconnect():
    print('disconnect')

@socketio.on('echo', namespace='/camera')
def echo(data: Any):
    print("echo ", data)
    emit('echo', data, namespace='/camera')

@socketio.on('echobin', namespace='/camera')
def get(data: Any):
    print("echobin", data)
    unpacked = msgpack.unpackb(bytearray(data["data"]))
    print("unpacked", unpacked)
    decoded = try_decode(unpacked)
    print("decoded", decoded)
    packed = msgpack.packb(decoded)
    print("packed", packed)
    emit('echobin', packed, namespace='/camera', binary=True)

def try_decode(data: Any)-> Any:
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
        self.delay = 0.
        self.thread_stop_event = threading.Event()
        super(CVThread, self).__init__()
        self.capture = cv2.VideoCapture(0) # http://docs.opencv.org/3.1.0/d8/dfe/classcv_1_1VideoCapture.html
        if self.capture.isOpened() is False:
            raise Exception("IO Error")

    def step(self):
        while not self.thread_stop_event.isSet():
            ret, frame = self.capture.read()
            if ret == False:
                print("ready frame!")
                return

            # print(type(frame), frame.shape, frame.ndim) # ex. <class 'numpy.ndarray'> (720, 1280, 3) 3
            ratio = 1/4 # 転送高速化のため画像を縮小
            size = (int(frame.shape[1]*ratio), int(frame.shape[0]*ratio))
            frame = cv2.resize(frame, size)
            print(frame.shape)

            # 画像を圧縮してバイト配列を得る            
            ret, frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            imgbuf = frame.tobytes('C')

            #print(type(imgbuf), len(imgbuf)) # ex. <class 'bytes'> 91557
            socketio.emit('broadcast', imgbuf, broadcast=True, namespace="/camera")
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

    socketio.run(app)
    print("end?")

