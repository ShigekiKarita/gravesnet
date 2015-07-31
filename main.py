import time
import argparse
import pickle
import sys

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
update_len = 1
eval_len = 8
n_epoch = 1000
use_gpu = args.gpu != -1
n_hidden = 100
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
    xi = (x[i] - mean) / stddev
    xe = context(numpy.concatenate((xi, [e[i]]), axis=1).astype(numpy.float32))
    xo = (x[i+1] - mean) / stddev
    t_x = context(numpy.array(xo).astype(numpy.float32))
    t_e = context(numpy.array([e[i + 1]]).astype(numpy.int32))
    return tuple(reshape2d(i) for i in (xe, t_x, t_e))


def set_volatile(state, volatile):
    for v in state.values():
        v.volatile = volatile
    return state


def evaluate(context, state, xs, es, mean, stddev):
    # TODO: init hidden-state (but use LSTM-state)
    state = set_volatile(state, True)
    total = mod.zeros(())
    indices = numpy.arange(len(es))
    numpy.random.shuffle(indices)
    total_seqlen = 0

    for i in indices[:eval_len]:
        x = xs[i]
        e = es[i]
        total_seqlen += len(e) - 1

        for t in range(len(es[i]) - 1):
            ci, cx, ce = create_inout(context, x, e, t, mean, stddev)
            state_ev, loss = model.forward_one_step(state, ci, cx, ce, train=False)
            total += loss.data.reshape(())

    state = set_volatile(state, False)
    t_loss = chainer.cuda.to_cpu(total)
    return t_loss / seq_len, t_loss / eval_len


if __name__ == '__main__':
    d = "/home/karita/Desktop/res_listed/"
    xs, es = load_dataset(d + "trainset_strokes.npy")
    txs, tes = load_dataset(d + "testset_v_strokes.npy")
    mean, stddev = pickle.load(open(d + "trainset_mean_std.npy", "rb"))
    print("load dataset")
    print("train", len(es), "test", len(tes))
    sys.stdout.flush()

    state = model.initial_state(mb_size, context)
    accum_loss = chainer.Variable(mod.zeros((), dtype=numpy.float32))
    rmsprop = chainer.optimizers.RMSpropGraves()
    rmsprop.setup(model.collect_parameters())
    total_loss = mod.zeros(())

    indices = numpy.arange(len(es))
    prev = time.time()
    n_point = 0
    loss_point_train = 0.0
    loss_seq_train = 0.0
    n_eval = 0
    for epoch in range(n_epoch):
        numpy.random.shuffle(indices)
        for i, n in zip(indices, range(len(es))):
            x = xs[i]
            e = es[i]
            seq_len = len(e)
            # TODO: init hidden-state
            for t in range(seq_len - 1):
                try:
                    inout = create_inout(context, x, e, t, mean, stddev)
                    state, loss_t = model.forward_one_step(state, *inout)
                    accum_loss += loss_t
                    total_loss += loss_t.data.reshape(())
                    n_point += 1
                except chainer.cuda.drv.MemoryError:
                    print("stop BPTT by out-of-memory")
                    break

            rmsprop.zero_grads()
            accum_loss.backward()
            accum_loss.unchain_backward()
            accum_loss = chainer.Variable(mod.zeros((), dtype=numpy.float32))
            rmsprop.update()

            if (n + 1) % update_len == 0:
                now = time.time()
                t_loss = chainer.cuda.to_cpu(total_loss)
                print('epoch {}, iter {}, loss/point: {:.6f}, loss/seq: {:.6f}, point/sec: {:.2f} '.format(
                    epoch, n,
                    t_loss / n_point,
                    t_loss / update_len,
                    float(n_point) / (now - prev)))
                sys.stdout.flush()
                loss_point_train += t_loss / n_point
                loss_seq_train += t_loss
                total_loss.fill(0)
                n_point = 0
                prev = now

            if (n + 1) % eval_len == 0:
                pickle.dump(model, open('model_%08d' % n_eval, 'wb'), -1)
                for k, v in state.items():
                    d = chainer.cuda.to_cpu(v.data)
                    pickle.dump(d, open('state_{}_{:08d}'.format(k, n_eval), 'wb'), -1)

                n_eval += 1
                print("eval-%08d" % n_eval)
                print('\ttrain: [loss/point: {:.6f}, loss/seq: {:.6f}]'.format(
                    loss_point_train / eval_len,
                    loss_seq_train / eval_len))
                sys.stdout.flush()
                loss_point, loss_seq = evaluate(context, state, txs, tes, mean, stddev)
                print('\ttest: [loss/point: {:.6f}, loss/seq: {:.6f}]'.format(loss_point, loss_seq))
                sys.stdout.flush()
                sys.stdout.flush()
                loss_point_train = 0.0
                loss_seq_train = 0.0
                prev = time.time()
