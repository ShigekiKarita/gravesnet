from chainer import cuda
from chainer import function


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

    def forward_cpu(self, inputs):
        return inputs,

    def forward_gpu(self, inputs):
        result = cuda.empty_like(inputs[0])
        cuda.elementwise(
            '''
            /* Gaussian output */
            float* r,
            /* Gaussian params */
            const float* w,
            const float* m1, const float* m2,
            const float* s1, const float* s2,
            const float* c,
            /* Gaussian input */
            const float* x1, const float* x2
            ''',
            '''
            float z1 = (*x1 - m1[i]) / s1[i];
            float z2 = (*x2 - m2[i]) / s2[i];
            float z3;

            z1 = pow(z1 - c[i] * z2, 2.0f);
            z2 = 1.0f - pow(c[i], 2.0f);
            z3 = 2.0f * 3.141592654f * s1[i] * s2[i] * sqrt(z2)
            r[i] = w[i] * exp(- z1 / (2.0f * z2)) / z3
            ''',
            'gaussian_mixture_2d_fwd'
        )(result, *inputs)
        return result,

    def backward_cpu(self, inputs, grad_outputs):
        return inputs,

    def backward_gpu(self, inputs, grad_outputs):
        gradients = (cuda.empty_like(i) for i in inputs[:-2]) # w/o (x1, x2)
        r = grad_outputs[0]
        cuda.elementwise(
            '''
            /* gradients */
            float* gw,
            float* gm1, float* gm2,
            float* gs1, float* gs2,
            float* gc,
            /* inputs */
            const float* w,
            const float* m1, const float* m2,
            const float* s1, const float* s2,
            const float* c,
            const float* x1, const float* x2, // scholar
            /* grad_outputs */
            const float* r
            ''',
            '''
            const float z1 = (*x1 - m1[i]) / s1[i];
            const float z2 = (*x2 - m2[i]) / s2[i];
            const float z3 = 1.0f / (1.0f - pow(c[i], 2.0f));
            const float z4 = pow(z1 - c[i] * z2, 2.0f);

            gw[i] = w[i] - r[i];
            gm1[i] = z3 / s1[i] * (z1 - c[i] * z2);
            gm2[i] = z3 / s2[i] * (z2 - c[i] * z1);
            gs1[i] = z3 * z1 * gm1[i] * s1[i] - 1.0f;
            gs2[i] = z3 * z2 * gm2[i] * s2[i] - 1.0f;
            gc[i]  = z1 * z2 + c[i] * (1.0f - z3 * z4);
            ''',
            'gaussian_mixture_2d_bwd'
        )(*gradients + inputs + grad_outputs)
        return gradients,


def gaussian_mixture_2d_reference(mix_weights, means, stddevs, correlations):
    """using chainer's automated propagation"""


def gaussian_mixture_2d(mix_weights1, mix_weights2, means1, means2,
                        stddevs1, stddevs2, correlations, positions):
    return GaussianMixture2D()(mix_weights1, mix_weights2, means1, means2,
                               stddevs1, stddevs2, correlations, positions)