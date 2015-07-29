import chainer
import chainer.functions as F
from chainer import cuda

from src.gaussian_mixture_2d import gaussian_mixture_2d
from src.gaussian_mixture_2d_ref import gaussian_mixture_2d_ref
from src.spilit_axis import split_axis_by_widths
from src.gradient_clip import gradient_clip
from src.sum_axis import sum_axis


def split_args(m, y, t_x, t_e):
    y_mixws, y_means, y_stdds, y_corrs, y_e = split_axis_by_widths(y, [m, 2 * m, 2 * m, m, 1])
    y_mixws = F.softmax(y_mixws)
    y_means0, y_means1 = split_axis_by_widths(y_means, 2)
    y_stdds0, y_stdds1 = split_axis_by_widths(F.exp(y_stdds), 2)
    y_corrs = F.tanh(y_corrs)

    t_x1, t_x2 = split_axis_by_widths(t_x, [1, 1])
    return (y_mixws, y_means0, y_means1, y_stdds0, y_stdds1, y_corrs, t_x1, t_x2), (y_e, t_e)


def concat_losses(p, e, t_e):
    loss_x = -F.sum(F.log(sum_axis(p))) / numpy.float32(p.data.shape[0])
    loss_e = F.sigmoid_cross_entropy(*e)
    return loss_x + loss_e


def loss_func(m, y, t_x, t_e):
    x, e = split_args(m, y, t_x, t_e)
    p = gaussian_mixture_2d(*x)
    return concat_losses(p, e, t_e)


class GravesPredictionNet(chainer.FunctionSet):
    """
    ref: sec. 4 in http://arxiv.org/abs/1308.0850
    """

    def __init__(self, nhidden=100, ngauss=30):
        ninput=3
        super(GravesPredictionNet, self).__init__(
            l1_first=F.Linear(ninput,  4 * nhidden, nobias=True),
            l1_recur=F.Linear(nhidden, 4 * nhidden),

            l2_first=F.Linear(ninput,  4 * nhidden, nobias=True),
            l2_recur=F.Linear(nhidden, 4 * nhidden),
            l2_input=F.Linear(nhidden, 4 * nhidden, nobias=True),

            l3_first=F.Linear(ninput,  4 * nhidden, nobias=True),
            l3_recur=F.Linear(nhidden, 4 * nhidden),
            l3_input=F.Linear(nhidden, 4 * nhidden, nobias=True),

            l4=F.Linear(nhidden * 3, 1 + ngauss * 6)
        )

    def initial_state(self, minibatch_size, context, train=True):
        state = dict()
        nhidden = self.l1_recur.W.shape[1]
        shape = (minibatch_size, nhidden)
        make_v = lambda : chainer.Variable(context(numpy.zeros(shape, dtype=numpy.float32)), volatile=not train)
        for n in range(1, 4):
            state.update(
                {
                    'h%s' % n: make_v(),
                    'c%s' % n: make_v()
                }
            )
        return state

    def forward_one_step(self, state, x_data, t_x_data, t_e_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t_x = chainer.Variable(t_x_data, volatile=not train)
        t_e = chainer.Variable(t_e_data, volatile=not train)

        h1_in = self.l1_first(x) + self.l1_recur(state['h1'])
        c1, h1 = F.lstm(state['c1'], h1_in)
        h1 = gradient_clip(h1, 10.0)
        h2_in = self.l2_first(x) + self.l2_recur(state['h2']) + self.l2_input(h1)
        c2, h2 = F.lstm(state['c2'], h2_in)
        h2 = gradient_clip(h2, 10.0)
        h3_in = self.l3_first(x) + self.l3_recur(state['h3']) + self.l3_input(h2)
        c3, h3 = F.lstm(state['c3'], h3_in)
        h3 = gradient_clip(h3, 10.0)

        y = self.l4(F.concat((h1, h2, h3)))
        y = gradient_clip(y, 100.0)
        n = int((y.data.shape[1] - 1) / 6)
        loss = loss_func(n, y, t_x, t_e)

        state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2, 'c3': c3, 'h3': h3}
        return state, loss


class GravesSynthesisNet(chainer.FunctionSet):
    """
    ref: sec. 5 in http://arxiv.org/abs/1308.0850
    """

    def __init__(self, ninput=3, nhidden=400, nlayer=3, noutput_gauss=30, nwindow_gauss=10):
        self.noutput_gauss = noutput_gauss
        self.nwindow_gauss = nwindow_gauss
        self.noutput = 1 + 4 * self.noutput_gauss

        super(GravesSynthesisNet, self).__init__(
            l1_first=F.Linear(ninput, nhidden, nobias=True),
            l1_recur=F.Linear(nhidden, nhidden),

            l2_first=F.Linear(ninput, nhidden, nobias=True),
            l2_recur=F.Linear(nhidden, nhidden),
            l2_input=F.Linear(nhidden, nhidden, nobias=True),

            l3_first=F.Linear(ninput, nhidden, nobias=True),
            l3_recur=F.Linear(nhidden, nhidden),
            l3_input=F.Linear(nhidden, nhidden, nobias=True),

            l4=F.Linear(nhidden * nlayer, self.noutput),

            # soft-window
            l1_w=F.Linear(1, nhidden, nobias=True),
            l2_w=F.Linear(1, nhidden, nobias=True),
            l3_w=F.Linear(1, nhidden, nobias=True),
            lw=F.Linear(nhidden, 3 * self.nwindow_gauss)
        )

    def forward_window_weight(self, a, b, k, u):
        w = k - u
        w = w * w * b
        return F.linear(a, F.exp(-w))

    def forward_window(self, a, b, k, cs):
        # FIXME: u is to be Number??
        u = chainer.Variable(range(len(cs)))
        if any(isinstance(i, cuda.GPUArray) for i in a):
            u.to_gpu()
        window_weights = self.forward_window_weight(a, b, k, u)
        return F.linear(window_weights, cs)

    def forward_one_step(self, x_data, c_data, y_data, state, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)
        c = chainer.Variable(c_data, volatile=not train)

        h1_in = self.l1_first(x) + self.l1_recur(state['h1']) + self.l1_w(state['w'])
        c1, h1 = F.lstm(state['c1'], h1_in)

        # soft window
        ws = F.exp(self.lw(h1))
        w_mixws, w_gains, w_means = split_axis_by_widths(ws, 3)
        w_means += state['w_means']
        w = self.forward_window(w_mixws, w_gains, w_means, c)

        h2_in = self.l2_first(x) + self.l2_recur(state['h2']) + self.l1_w(w) + self.l2_input(h1)
        c2, h2 = F.lstm(state['c2'], h2_in)

        h3_in = self.l3_first(x) + self.l3_recur(state['h3']) + self.l1_w(w) + self.l3_input(h2)
        c3, h3 = F.lstm(state['c3'], h3_in)

        y = self.l4(F.concat(h1, h2, h3))

        state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2, 'c3': c3, 'h3': h3,
                 'w': w, 'w_means': w_means}
        return state, loss_func(self.noutput_gauss, y, t)


import numpy


def forward(model, x_list):
    h = chainer.Variable(numpy.zeros(model.state_size, dtype=numpy.float32))
    loss = 0.0
    for current, next in zip(x_list, x_list[1:]):
        h, new_loss = model.forward_one_step(h, current, next)
        loss += new_loss
    return loss

from chainer import optimizers

def train(model, x_list):
    opt = optimizers.RMSpropGraves()