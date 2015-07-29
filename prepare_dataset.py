import sys
import pickle

from compute_meanstd import xs
from tool.dataset import parse_IAMdataset_strokes

if __name__ == '__main__':
    d = "/home/karita/Documents/IAM/"
    files = ["testset_v", "testset_t", "testset_f", "trainset"]

    f = sys.argv[1]
    xs, es = parse_IAMdataset_strokes(d + "task1/" + f + ".txt", d + "data/lineStrokes")
    pickle.dump((xs, es), open(f + "_strokes.npy", "wb"), -1)

