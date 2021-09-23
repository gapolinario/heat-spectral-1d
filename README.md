# Pseudospectral Simulation of the 1D Heat Equation

Here is some code I wrote for the Heat equation in 1D

The [blog post][blog1] ran with the following parameters:
`time seq 0 199 | xargs -I{} -P 7 ./heat_1d.x {} 9 1000000 .1 .01 1. >/dev/null`

`heat_inplace_1d.c` does the same as `heat_1d.c` but with inplace FFTs (see [FFTW documentation][fftinp])

`heat_jentzen_1d.c` applies the algorithm of:

Arnulf Jentzen, Peter Kloeden, and Georg Winkel. "Efficient simulation of nonlinear parabolic SPDEs with additive noise." The Annals of Applied Probability 21.3 (2011): 908-950.

The simulation of the Ornstein-Uhlenbeck equation with an exact algorithm is shown in the `ornstein-uhlenbeck.ipynb` file. This is applied to the Heat equation (and to nonlinear equations) in the Jentzen-Kloeden-Winkel algorithm. The random variables must be normal random variables in this case. In the predictor-corrector algorithm, I used a uniform random variable with unit variance.

RNG for standard normal random variables was slightly modified at p=0 and p=1, use a different algorithm for serious applications

[blog1]: https://gapolinario.github.io/blog/2021/time-scales-spectral-simulation-heat-equation/
[fftinp]: http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html
