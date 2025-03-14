import os
from pprint import pprint

"""
28apr2023, TT: utiltiy functions
"""

def get_class_names(data_dir):
    """ Get Dict of, class names and class name index from data dir """
    class_names = [name for name in os.listdir(data_dir)]
    class_name_dict ={}
    for i,name in enumerate(class_names):
        class_name_dict[i] = name

    return class_name_dict

#testing
if __name__ == '__main__':

    dataset_dir = "datasets/Vegetable/test/"
    pprint(list(get_class_names(dataset_dir).values()))
