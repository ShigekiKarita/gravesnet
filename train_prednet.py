import time
import math
import pickle

import chainer
import chainer.cuda
import chainer.optimizers

import six
import numpy

from src.dataset import parse_IAMxml
from src.gravesnet import GravesPredictionNet

range = six.moves.range

model = GravesPredictionNet()
mod = chainer.cuda
context = lambda x: x
if chainer.cuda.available:
    chainer.cuda.init()
    model.to_gpu()
    context = chainer.cuda.to_gpu

def mini_batch(mb_size, storage, index):
    xs_size = xs.shape[0]
    jump = xs_size // mb_size
    return numpy.array([storage[(jump * j + index) % xs_size] for j in range(mb_size)])

if __name__ == '__main__':
    xs, es = parse_IAMxml("res/strokesz.xml")
    t = 0
    mb_size = 8
    n_hidden = 100
    bp_len = 100
    update_len = 1000
    n_epoch = 1000

    state = model.initial_state((mb_size, n_hidden), context, numpy)
    accum_loss = chainer.Variable(mod.zeros((), dtype=numpy.float32))
    rmsprop = chainer.optimizers.RMSpropGraves()
    rmsprop.setup(model.collect_parameters())
    total_loss = mod.zeros(())
    prev = time.time()

    train_data = numpy.concatenate((xs, es),axis=1)
    es = es.astype(numpy.int32)
    jump = xs.shape[0] // mb_size

    for i in six.moves.range(jump * n_epoch):
        xe_batch = context(numpy.array(mini_batch(mb_size, train_data, i)))
        t_x_batch = context(numpy.array(mini_batch(mb_size, xs, i + 1)))
        t_e_batch = context(numpy.array(mini_batch(mb_size, es, i + 1)))
        state, loss_i = model.forward_one_step(xe_batch, t_x_batch, t_e_batch, state)
        accum_loss += loss_i
        total_loss += loss_i.data.reshape(())

        if (i + 1) % bp_len == 0:
            rmsprop.zero_grads()
            accum_loss.backward()
            accum_loss.unchain_backward()
            accum_loss = chainer.Variable(mod.zeros((), dtype=numpy.float32))
            rmsprop.update()

        if (i + 1) % update_len == 0:
            now = time.time()
            throuput = float(update_len) / (now - prev)
            avg_loss = math.exp(chainer.cuda.to_cpu(total_loss) / update_len)
            print('iter {} training loss: {:.2f} ({:.2f} iters/sec)'.format(
                i + 1, avg_loss, throuput))
            prev = now
            total_loss.fill(0)
            # pickle.dump(model, open('model%04d' % (i+1), 'wb'), -1)