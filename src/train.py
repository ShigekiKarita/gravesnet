import argparse
import pickle
import sys
import time

import chainer
import chainer.cuda
import chainer.optimizers
import numpy


class OptimizationSizes(object):
    def __init__(self,
                 epoch_size=1000, train_size=1,
                 eval_size=8, mini_batch_size=1):
        self.epoch = epoch_size
        self.train = train_size
        self.eval = eval_size
        self.mini_batch = mini_batch_size


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
    return numpy.array(
        [xs[(jump * j + index) % xs_size] for j in range(mb_size)]
    )


def reshape2d(x):
    return x.reshape(1, len(x))


def create_inout(context, x, e, i, mean, stddev):
    xi = (x[i] - mean) / stddev
    xe = context(numpy.concatenate((xi, [e[i]]), axis=1).astype(numpy.float32))
    xo = (x[i+1] - mean) / stddev
    t_x = context(numpy.array(xo).astype(numpy.float32))
    t_e = context(numpy.array([e[i + 1]]).astype(numpy.int32))
    return tuple(reshape2d(i) for i in (xe, t_x, t_e))


def set_volatile(lstm_cells, volatile):
    for v in lstm_cells.values():
        v.volatile = volatile
    return lstm_cells


def evaluate(context, model, lstm_cells: chainer.Variable,
             sizes: OptimizationSizes, xs, es, mean, stddev):
    set_volatile(lstm_cells, True)
    total = numpy.zeros(())
    indices = numpy.arange(len(es))
    numpy.random.shuffle(indices)
    total_seq_len = 0

    for i in indices[:sizes.eval]:
        x = xs[i]
        e = es[i]
        total_seq_len += len(e) - 1
        hidden_state = model.initial_state(1, context, "h", train=False)
        for t in range(len(es[i]) - 1):
            ci, cx, ce = create_inout(context, x, e, t, mean, stddev)
            hidden_state, lstm_cells, loss = model.forward_one_step(
                hidden_state, lstm_cells, ci, cx, ce, train=False
            )
            total += loss.data.reshape(())

    set_volatile(lstm_cells, False)
    t_loss = chainer.cuda.to_cpu(total)
    return t_loss / total_seq_len, t_loss / sizes.eval


def optimize(model, sizes: OptimizationSizes, data_dir: str):
    xs, es = load_dataset(data_dir + "trainset_strokes.npy")
    txs, tes = load_dataset(data_dir + "testset_v_strokes.npy")
    mean, stddev = pickle.load(open(data_dir + "trainset_mean_std.npy", "rb"))
    print("load dataset")
    print("train", len(es), "test", len(tes))
    sys.stdout.flush()

    args = parse_args()
    context = lambda x: x
    if chainer.cuda.available and args.gpu != -1:
        print("use gpu")
        chainer.cuda.init(args.gpu)
        model.to_gpu()
        context = chainer.cuda.to_gpu
    else:
        print("use cpu")

    lstm_cells = model.initial_state(sizes.mini_batch, context, "c")
    rmsprop = chainer.optimizers.RMSpropGraves()
    rmsprop.setup(model.collect_parameters())
    total_loss = context(numpy.zeros(()))

    indices = numpy.arange(len(es))
    prev = time.time()
    n_point = 0
    loss_point_train = 0.0
    loss_seq_train = 0.0
    n_eval = 0
    for epoch in range(sizes.epoch):
        numpy.random.shuffle(indices)
        for i, n in zip(indices, range(len(es))):
            x = xs[i]
            e = es[i]
            seq_len = len(e)
            hidden_state = model.initial_state(sizes.mini_batch, context, "h")
            accum_loss = chainer.Variable(
                context(numpy.zeros((), dtype=numpy.float32))
            )
            for t in range(seq_len - 1):
                inout = create_inout(context, x, e, t, mean, stddev)
                hidden_state, lstm_cells, loss_t = model.forward_one_step(
                    hidden_state, lstm_cells, *inout
                )
                accum_loss += loss_t
                total_loss += loss_t.data.reshape(())
                n_point += 1

            if (n + 1) % sizes.train == 0:
                rmsprop.zero_grads()
                accum_loss.backward()
                accum_loss.unchain_backward()
                rmsprop.update()

                now = time.time()
                t_loss = chainer.cuda.to_cpu(total_loss)
                print(
                    'epoch {}, iter {}, loss/point: {:.6f}, loss/seq: {:.6f}, point/sec: {:.2f} '.format(
                        epoch, n,
                        t_loss / n_point,
                        t_loss / sizes.train,
                        float(n_point) / (now - prev)))
                sys.stdout.flush()
                loss_point_train += t_loss / n_point
                loss_seq_train += t_loss
                total_loss.fill(0)
                n_point = 0
                prev = now

            if (n + 1) % sizes.eval == 0:
                pickle.dump(model, open('model_%08d' % n_eval, 'wb'), -1)
                for k, v in lstm_cells.items():
                    d = chainer.cuda.to_cpu(v.data)
                    pickle.dump(
                        d, open('lstm_{}_{:08d}'.format(k, n_eval), 'wb'), -1
                    )

                n_eval += 1
                print("eval-%08d" % n_eval)
                print('\ttrain: [loss/point: {:.6f}, loss/seq: {:.6f}]'.format(
                    loss_point_train / sizes.eval,
                    loss_seq_train / sizes.eval))
                sys.stdout.flush()
                lstm_copy = lstm_cells.copy()
                loss_point, loss_seq = evaluate(
                    context, model, lstm_copy, sizes, txs, tes, mean, stddev
                )
                print(
                    '\ttest:  [loss/point: {:.6f}, loss/seq: {:.6f}]'.format(
                        loss_point, loss_seq
                    )
                )
                sys.stdout.flush()
                loss_point_train = 0.0
                loss_seq_train = 0.0
                prev = time.time()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Trained model')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    return args
