from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy


class GaussianMixture2D(function.Function):

    """
    2-Dimensional Gaussian Mixture Model (GMM) Layer
    forward(inputs):
        inputs:
            GMM-parameters (output signals from previous layer):
                w: 1D mixture weights
                m1, m2: 2D Gaussian means
                s1, s2: 2D Gaussian standard derivatives
                c: 1D Gaussian correlations

            GMM-position (teach signals):
                x1, x2: 2D position in pdf

        return: weighted Gaussian probabilities as w[i] * Pr(teach|params[i])

    backward(inputs, grad_outputs):
        inputs:
            forward inputs: GMM-params + GMM-position

        grad_outputs:
            forward outputs: weighted Gaussian probabilities

        return: gradient of GMM-params
    """

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 8)
        mbsize = in_types[0].shape[0]
        type_check.expect(
            in_types[0].dtype == in_types[1].dtype,
            # check if the mini_batch sizes are same
            in_types[0].shape[0] == in_types[1].shape[0]
        )


    def forward_cpu(self, inputs):
        w, m1, m2, s1, s2, c, x1, x2 = inputs
        z1 = (x1 - m1) / s1
        z2 = (x2 - m2) / s2
        z1 = (z1 - c * z2)**2
        z2 = 1.0 - c**2
        z3 = 2.0 * numpy.pi * s1 * s2 * numpy.sqrt(z2)
        self.y = w * numpy.exp(- z1 / (2.0 * z2)) / z3
        return self.y,

    def forward_gpu(self, inputs):
        self.y = cuda.empty_like(inputs[0])
        cuda.elementwise(
            '''
            float* r,
            const int ngauss,
            const float* w,
            const float* m1, const float* m2,
            const float* s1, const float* s2,
            const float* c,
            const float* x1, const float* x2
            ''',
            '''
            const int j = i / ngauss;
            float z1 = (x1[j] - m1[i]) / s1[i];
            float z2 = (x2[j] - m2[i]) / s2[i];
            float z3;

            z1 = pow(z1 - c[i] * z2, 2.0f);
            z2 = 1.0f - pow(c[i], 2.0f);
            z3 = 2.0f * 3.141592654f * s1[i] * s2[i] * sqrt(z2);
            r[i] = w[i] * exp(- z1 / (2.0f * z2)) / z3;
            ''',
            'gaussian_mixture_2d_fwd'
        )(self.y, self.y.shape[1], *inputs)
        return self.y,

    def backward_cpu(self, inputs, grad_outputs):
        w, m1, m2, s1, s2, c, x1, x2 = inputs

        ysum = self.y.sum(1).reshape(self.y.shape[0], 1)
        self.y = - self.y / ysum
        gw = w + self.y

        z1 = (x1 - m1) / s1
        z2 = (x2 - m2) / s2
        z3 = 1.0 / (1.0 - c**2.0)
        z4 = (z1 - c * z2)**2.0

        gm1 = z3 / s1 * (z1 - c * z2)
        gm2 = z3 / s2 * (z2 - c * z1)
        gs1 = (x1 - m1) * gm1 - 1.0
        gs2 = (x2 - m2) * gm2 - 1.0
        gc = z1 * z2 + c * (1.0 - z3 * z4)

        gt = tuple(self.y * g for g in (gm1, gm2, gs1, gs2, gc))
        return (gw,) + gt + (None, None)

    def backward_gpu(self, inputs, grad_outputs):
        ngauss = self.y.shape[1]
        gradients = tuple(cuda.empty_like(i) for i in inputs[:-2]) # w/o (x1, x2)
        args = gradients + inputs
        ysum = None
        with cuda.using_cumisc():
            ysum = cuda.cumisc.sum(self.y, 1).reshape((self.y.shape[0], 1))
        cuda.elementwise(
            '''
            const float* r,
            const float* rsum,
            const int ngauss,
            float* gw,
            float* gm1, float* gm2,
            float* gs1, float* gs2,
            float* gc,
            const float* w,
            const float* m1, const float* m2,
            const float* s1, const float* s2,
            const float* c,
            const float* x1, const float* x2
            ''',
            '''
            const int j = i / ngauss;
            const float z1 = (x1[j] - m1[i]) / s1[i];
            const float z2 = (x2[j] - m2[i]) / s2[i];
            const float z3 = 1.0f / (1.0f - pow(c[i], 2.0f));
            const float z4 = pow(z1 - c[i] * z2, 2.0f);
            const float z5 = - r[i] / rsum[j];

            gw[i] = w[i] + z5;
            gm1[i] = z3 / s1[i] * (z1 - c[i] * z2);
            gm2[i] = z3 / s2[i] * (z2 - c[i] * z1);
            gs1[i] = (x1[j] - m1[i]) * gm1[i] - 1.0f;
            gs2[i] = (x2[j] - m2[i]) * gm2[i] - 1.0f;
            gc[i]  = z1 * z2 + c[i] * (1.0f - z3 * z4);

            gm1[i] *= z5;
            gm2[i] *= z5;
            gs1[i] *= z5;
            gs2[i] *= z5;
            gc[i]  *= z5;
            ''',
            'gaussian_mixture_2d_bwd'
        )(self.y, ysum, ngauss, *args)
        # tuple(- self.y * g for g in gradients[1:])
        return gradients + (None, None)  # for target signals


def gaussian_mixture_2d(
        mix_weights, means1, means2,
        stddevs1, stddevs2, correlations,
        position1, position2):
    return GaussianMixture2D()(
        mix_weights, means1, means2,
        stddevs1, stddevs2, correlations,
        position1, position2)

