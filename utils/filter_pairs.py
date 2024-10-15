from utils import filterPair


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]
