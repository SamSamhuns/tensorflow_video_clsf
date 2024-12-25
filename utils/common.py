import glob
import json
import functools
from pathlib import Path
from functools import partial
from collections import OrderedDict


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def init_obj(config, name, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and
    returns the instance initialized with corresponding arguments given.

    `object = config.init_obj('name', module, a, b=1)`
    is equivalent to
    `object = module.name(a, b=1)`
    """
    module_name = config[name]['type']
    module_args = dict(config[name]['args'])
    assert all([k not in module_args for k in kwargs]
               ), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)


def init_ftn(config, name, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    function with given arguments fixed with functools.partial.

    `function = config.init_ftn('name', module, a, b=1)`
    is equivalent to
    `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
    """
    module_name = config[name]['type']
    module_args = dict(config[name]['args'])
    assert all([k not in module_args for k in kwargs]
               ), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return partial(getattr(module, module_name), *args, **module_args)


def rgetattr(obj, attr, *args):
    """
    recursively get attrs. i.e. rgetattr(module, "sub1.sub2.sub3")
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def recursively_get_file_paths(root_dir, ext="npy"):
    """
    Get file paths recursively in the 'root_dir' with the extension 'ext'
    """
    fpaths = []
    for fpath in glob.glob(f"{root_dir}/**/*.{ext}", recursive=True):
        fpaths.append(fpath)
    return fpaths


def validate_data_paths(data_paths, data_type="train"):
    """
    Validate the data paths
    """
    if not data_paths or len(data_paths) == 0:
        raise FileNotFoundError(
            f"No data found for {data_type}. Please check the data dir or file paths.")
    print(f"Found {len(data_paths)} {data_type} samples.")
