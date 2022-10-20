from numpy import ndarray, linspace, sqrt, cumprod
from numpy.random import randn


class NoiseScheduler:
    def __init__(self, T: int, start: float = 0.0001, end: float = 0.02) -> None:
        """ Pre-calculate values that can be reused when noising. """
        self.betas = linspace(start, end, T)
        self.alphas = 1. - self.betas
        self.cumprod_alphas = cumprod(self.alphas)
        self.sqrt_cumprod_alphas = sqrt(self.cumprod_alphas)
        self.sqrt_one_minus_cumprod_alphas = sqrt(1. - self.cumprod_alphas)

    def forward(self, x0: ndarray, t: int) -> ndarray:
        """ 
            xt = sqrt(cumprod_alpha_t) * x0 + sqrt(1 - cumprod_alpha_t) * noise

            Use pre-calculated values along with the input and noise of the input shape
            to construct the noise of x0 at timestep t.

            This allows us to do one noise step from 0 to t instead of many iterations.
        """
        return self.sqrt_cumprod_alphas[t] * x0 + self.sqrt_one_minus_cumprod_alphas[t] * randn(*x0.shape)
