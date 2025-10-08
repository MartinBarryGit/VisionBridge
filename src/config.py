import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(parent_dir)
data_dir = os.path.join(grandparent_dir, 'dataset')