import cloudpickle as pickle
with open("/tmp/data.pkl", "r") as f: unpickled = pickle.load(f)
space = unpickled["search"]
