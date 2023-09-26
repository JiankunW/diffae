import yaml
import torch
import lmdb


def load_yaml(filename):
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

# label: B  return: B x N
def get_one_hot(label, N):
    size=list(label.size())
    size.append(N)

    ones=torch.eye(N) # one-hot to be selected
    label=label.view(-1)
    output=ones.index_select(0,label)
    return output.view(*size)

def open_lmdb(path):
    env = lmdb.open(
        path,
        max_readers=32,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    return env.begin(write=False)
