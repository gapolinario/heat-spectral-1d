run in parallel, one line
no output on command line
`time seq 0 99 | xargs -I{} -P 7 ./heat_1d.x {} 8 100 .1 .01 1. >/dev/null`

1. float?
