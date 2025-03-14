import pickle
import os

"""
18apr2023, TT: utility functions for saving , loading model files in pickle format
"""

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

def save_model(model,name,path):
    check_path(path)
    filename = os.path.join(path,name)
    with open(filename,"wb") as f:
        pickle.dump(model,f)

    print("Model {} saved to {}".format(name,path))

def load_model(name,path):
    filename = os.path.join(path,name)
    with open(filename,"rb") as f:
        model = pickle.load(f)

    return model
