import time
import argparse
import pickle

import chainer
import chainer.cuda
import chainer.optimizers
import six
import numpy

from src.gravesnet import GravesPredictionNet

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='res/model', type=str,
                    help='Trained model')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()


mb_size = 1
update_len = 1000
eval_len = 10000
n_epoch = 1000
use_gpu = args.gpu != -1
n_hidden = 400
model = GravesPredictionNet(n_hidden)

range = six.moves.range
mod = numpy
context = lambda x: x

if chainer.cuda.available and use_gpu:
    print("use gpu")
    chainer.cuda.init(args.gpu)
    model.to_gpu()
    mod = chainer.cuda
    context = chainer.cuda.to_gpu


def merge_strokes_in_line(lines, elem_type):
    return [numpy.concatenate(l).astype(elem_type) for l in lines]


def load_dataset(path):
    x, e = pickle.load(open(path, 'rb'))
    x = merge_strokes_in_line(x, numpy.float32)
    e = merge_strokes_in_line(e, numpy.int32)
    return x, e


def mini_batch(mb_size, xs, index):
    xs_size = xs.shape[0]
    jump = xs_size // mb_size
    return numpy.array([xs[(jump * j + index) % xs_size] for j in range(mb_size)])


def reshape2d(x):
    return x.reshape(1, len(x))


def create_inout(context, x, e, i, mean, stddev):
    xs = (x - mean) / stddev
    xe = context(numpy.concatenate((xs[i], [e[i]]), axis=1).astype(numpy.float32))
    t_x = context(numpy.array(xs[i + 1]).astype(numpy.float32))
    t_e = context(numpy.array([e[i + 1]]).astype(numpy.int32))
    return tuple(reshape2d(i) for i in (xe, t_x, t_e))


def evaluate(xs, es, context):
    total = mod.zeros(())
    state = model.initial_state(1, context, False)
    m = 1000
    n = numpy.random.randint(0, 100000)
    for i in range(n, n + m):
        x = context(numpy.concatenate((xs[i], es[i]), axis=1).reshape(1, 3).astype(numpy.float32))
        tx = context(xs[i+1]).reshape(1, 2)
        te = context(es[i+1]).reshape(1, 1)
        # inout = create_inout(1, context, xs, es, i)
        state, loss = model.forward_one_step(state, x, tx, te, train=False)
        total += loss.data.reshape(())
    return chainer.cuda.to_cpu(total) / m


if __name__ == '__main__':
    d = "/home/karita/Desktop/res_listed/"
    xs, es = load_dataset(d + "trainset_strokes.npy")
    txs, tes = load_dataset(d + "testset_v_strokes.npy")
    mean, stddev = pickle.load(open(d + "trainset_mean_std.npy", "rb"))
    print("load dataset")
    print("train", len(es), "test", len(tes))

    state = model.initial_state(mb_size, context)
    accum_loss = chainer.Variable(mod.zeros((), dtype=numpy.float32))
    rmsprop = chainer.optimizers.RMSpropGraves()
    rmsprop.setup(model.collect_parameters())
    total_loss = mod.zeros(())
    prev = time.time()
    indices = numpy.arange(len(es))
    for epoch in range(n_epoch):
        numpy.random.shuffle(indices)
        for n in indices:
            x = xs[n]
            e = es[n]
            seq_len = len(e)
            for t in range(seq_len - 1):
                inout = create_inout(context, x, e, t, mean, stddev)
                state, loss_t = model.forward_one_step(state, *inout)
                accum_loss += loss_t
                total_loss += loss_t.data.reshape(())

            rmsprop.zero_grads()
            accum_loss.backward()
            accum_loss.unchain_backward()
            accum_loss = chainer.Variable(mod.zeros((), dtype=numpy.float32))
            rmsprop.update()

            if (n + 1) % update_len == 0:
                now = time.time()
                throuput = float(update_len) / (now - prev)
                average_loss = chainer.cuda.to_cpu(total_loss) / update_len
                print('iter {} training loss: {:.6f} ({:.2f} iters/sec)'.format(
                    t + 1, average_loss, throuput))
            if (n + 1) & eval_len == 0:
                ev_loss = evaluate(txs, tes, context)
                print('test loss: {}'.format(ev_loss))
                prev = now
                total_loss.fill(0)
            # pickle.dump(model, open('model%04d' % (i+1), 'wb'), -1)