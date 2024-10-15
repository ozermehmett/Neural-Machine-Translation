from utils import normalizeString


def load_dataset(path):
  with open(path,'r') as f:
    lines = f.readlines()

  pairs = [[normalizeString(line.split('\t')[0]), normalizeString(line.split('\t')[1])] for line in lines]
  return pairs
