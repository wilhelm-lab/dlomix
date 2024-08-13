import pickle

import numpy as np


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def convert_nested_list_to_numpy_array(nested_list, dtype=np.float32):
    return np.array([np.array(x, dtype=dtype) for x in nested_list])


def lower_and_trim_strings(strings):
    return [s.lower().trim() for s in strings]


def get_constructor_call_object_creation(object_instance):
    members = [
        attr
        for attr in vars(object_instance)
        if not callable(getattr(object_instance, attr))
        and not attr.startswith(("_", "__"))
    ]
    values = [object_instance.__getattribute__(m) for m in members]

    repr_str = ", ".join([f"{m}={v}" for m, v in zip(members, values)])

    return f"{object_instance.__class__.__name__}({repr_str})"


def flatten_dict_for_values(d):
    if not isinstance(d, dict):
        return d
    else:
        items = []
        for v in d.values():
            if isinstance(v, dict):
                return flatten_dict_for_values(v)
            else:
                items.append(v)
        return items
