run in parallel, one line, no output on command line
[blog post][blog1] ran with the following parameters:
`time seq 0 199 | xargs -I{} -P 7 ./heat_1d.x {} 9 1000000 .1 .01 1. >/dev/null`



[blog1]: https://gapolinario.github.io/blog/2021/time-scales-spectral-simulation-heat-equation/
