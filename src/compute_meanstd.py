import sys
import pickle
import numpy

f = sys.argv[1]
xs, _ = pickle.load(open(f + "_strokes.npy", "rb"))
xs = [numpy.concatenate(i) for i in xs]
xs = numpy.concatenate(xs)
m = numpy.mean(xs, 0)
s = numpy.std(xs, 0)
pickle.dump((m, s), open(f + "_mean_std.npy", "wb"), -1)