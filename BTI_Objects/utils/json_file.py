import json
import numpy as np

"""
18apr2023, TT: utility functions to read and save json format file
"""

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    """https://stackoverflow.com/questions/27909658/json-encoder-and-decoder-for-complex-numpy-arrays"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def to_json_file(var,filename):
    assert filename, "not filename given"
    with open(filename,'w') as f:
        json.dump(var,f,cls=NumpyEncoder)


def from_json_file(filename):
    with open(filename,'r') as f:
        data = json.load(f)
    return data