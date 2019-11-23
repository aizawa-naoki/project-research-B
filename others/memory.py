import sys
from itertools import chain
from collections import deque


def compute_object_size(o, handlers={}):
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    # estimate sizeof object without __sizeof__
    default_size = sys.getsizeof(0)

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)
        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s
    return sizeof(o)


def show_objects_size(threshold, unit=2):
    disp_unit = {0: 'bites', 1: 'KB', 2: 'MB', 3: 'GB'}
    # 処理中に変数が変動しないように固定
    globals_copy = globals().copy()
    for object_name in globals_copy.keys():
        size = compute_object_size(eval(object_name))
        if size > threshold:
            print('{:<15}{:.3f} {}'.format(object_name, size, disp_unit[unit]))
