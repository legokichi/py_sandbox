from typing import no_type_check, no_type_check_decorator
from typing import get_type_hints, cast
from typing import TypeVar, NewType, Type, Generic#, ClassVar

from typing import Container, Hashable, Iterable, Sized, Callable, Awaitable, AsyncIterable
from typing import Iterator, Sequence, Set, Mapping, MappingView, AsyncIterator#, Coroutine
from typing import Generator, MutableSequence, ByteString, MutableSet, MutableMapping, ItemsView, KeysView, ValuesView

from typing import SupportsInt, SupportsFloat, SupportsAbs, SupportsRound
from typing import Union, Optional, AbstractSet, Reversible
from typing import Any, re, io, AnyStr, Tuple, NamedTuple, List, Dict, DefaultDict 

import dlib
from skimage import io
from PIL import Image
import numpy as np
import cv2

image_file = 'capture.png' # type: string
img = io.imread(image_file) # type: np.ndarray
# http://kivantium.hateblo.jp/entry/2015/07/25/184346
# selective search
rects = [] # type: List<(bool, {left:Callable[[],int], top:Callable[[],int], right:Callable[[],int], bottom:Callable[[],int]})>
dlib.find_candidate_object_locations(img, rects, min_size=1000)

win = dlib.image_window() # type: dlib.image_window
win.set_image(img)
for k, d in enumerate(rects):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    win.add_overlay(d)
dlib.hit_enter_to_continue()
