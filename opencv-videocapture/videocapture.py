from typing import no_type_check, no_type_check_decorator
from typing import get_type_hints, cast
from typing import TypeVar, NewType, Type, Generic#, ClassVar

from typing import Container, Hashable, Iterable, Sized, Callable, Awaitable, AsyncIterable
from typing import Iterator, Sequence, Set, Mapping, MappingView, AsyncIterator#, Coroutine
from typing import Generator, MutableSequence, ByteString, MutableSet, MutableMapping, ItemsView, KeysView, ValuesView

from typing import SupportsInt, SupportsFloat, SupportsAbs, SupportsRound
from typing import Union, Optional, AbstractSet, Reversible
from typing import Any, re, io, AnyStr, Tuple, NamedTuple, List, Dict, DefaultDict 
 
from PIL import Image
import numpy as np
import cv2
 

mirror = True # type: bool
size   = None # type: Optional[(int, int)]

cap = cv2.VideoCapture(0) # type: cv2.VideoCapture

while True:
    ret, frame = cap.read() # type: (bool, np.ndarray)

    if not ret: continue


    # 鏡のように映るか否か
    if mirror is True:
        frame = frame[:,::-1]

    # フレームをリサイズ
    # sizeは例えば(800, 600)
    if size is not None and len(size) == 2:
        frame = cv2.resize(frame, size)

    cv2.imshow('camera capture', frame)

    # 1msec待ってESCキーで画像保存
    k = cv2.waitKey(1) # type: int
    if k == 27:
        pilImg = Image.fromarray(np.uint8(frame)) # type: Image
        pilImg.save('capture.png', "png")
        break

cap.release()
cv2.destroyAllWindows()