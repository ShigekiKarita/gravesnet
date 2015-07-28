import time
import math
import pickle

import chainer
import chainer.cuda
import chainer.optimizers

import six
import numpy

from src.gravesnet import GravesPredictionNet


mb_size = 64
bp_len = 100
update_len = 1000
n_epoch = 1000
use_gpu = True
n_hidden = 400
model = GravesPredictionNet(n_hidden)


range = six.moves.range
mod = numpy
context = lambda x: x
if chainer.cuda.available and use_gpu:
    print("use gpu")
    chainer.cuda.init()
    model.to_gpu()
    mod = chainer.cuda
    context = chainer.cuda.to_gpu


def load_dataset(path):
    xs, es = pickle.load(open(path, 'rb'))
    return numpy.float32(xs), numpy.int32(es)


def mini_batch(mb_size, xs, index):
    xs_size = xs.shape[0]
    jump = xs_size // mb_size
    return numpy.array([xs[(jump * j + index) % xs_size] for j in range(mb_size)])


def create_inout(mb_size, context, xs, es, i):
    x_batch = mini_batch(mb_size, xs, i)
    e_batch = mini_batch(mb_size, es, i)
    xe_batch = context(numpy.concatenate((x_batch, e_batch), axis=1).astype(numpy.float32))
    t_x_batch = context(mini_batch(mb_size, xs, i + 1))
    t_e_batch = context(mini_batch(mb_size, es, i + 1))
    return xe_batch, t_x_batch, t_e_batch


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
    xs, es = load_dataset("res/trainset.npy")
    txs, tes = load_dataset("res/testset_v.npy")
    print("train", es.shape, "test", tes.shape)

    print("load dataset")

    state = model.initial_state(mb_size, context)
    accum_loss = chainer.Variable(mod.zeros((), dtype=numpy.float32))
    rmsprop = chainer.optimizers.RMSpropGraves()
    rmsprop.setup(model.collect_parameters())
    total_loss = mod.zeros(())
    prev = time.time()
    jump = xs.shape[0] // mb_size

    for i in six.moves.range(jump * n_epoch):
        inout = create_inout(mb_size, context, xs, es, i)
        state, loss_i = model.forward_one_step(state, *inout)
        accum_loss += loss_i
        total_loss += loss_i.data.reshape(())

        if (i + 1) % bp_len == 0:
            rmsprop.zero_grads()
            accum_loss.backward()
            accum_loss.unchain_backward()
            accum_loss = chainer.Variable(mod.zeros((), dtype=numpy.float32))
            rmsprop.update()

        if (i + 1) % update_len == 0:
            ev_loss = evaluate(txs, tes, context)
            now = time.time()
            throuput = float(update_len) / (now - prev)
            avg_loss = chainer.cuda.to_cpu(total_loss) / update_len
            print('iter {} training loss: {:.6f} ({:.2f} iters/sec)'.format(
                i + 1, avg_loss, throuput))
            print('test loss: {}'.format(ev_loss))
            prev = now
            total_loss.fill(0)

            # pickle.dump(model, open('model%04d' % (i+1), 'wb'), -1)