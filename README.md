run in parallel, one line, no output on command line
[blog post][blog1] ran with the following parameters:
`time seq 0 199 | xargs -I{} -P 7 ./heat_1d.x {} 9 1000000 .1 .01 1. >/dev/null`

`heat_inplace_1d.c` does the same as `heat_1d.c` but with inplace FFTs (see [FFTW documentation][fftinp])

`heat_jentzen_1d.c` applies the algorithm of:

Arnulf Jentzen, Peter Kloeden, and Georg Winkel. "Efficient simulation of nonlinear parabolic SPDEs with additive noise." The Annals of Applied Probability 21.3 (2011): 908-950.

[blog1]: https://gapolinario.github.io/blog/2021/time-scales-spectral-simulation-heat-equation/
[fftinp]: http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html
