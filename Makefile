WARN = -Wmissing-prototypes -Wall #-Winline
#WARN = -Wmissing-prototypes -Wall -W
OPTI = -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800
#--param inline-unit-growth=500 --param large-function-growth=900 #for gcc 4
#OPTI = /Ot /Ob2 /Oy /Ox /Oi /GL /G6
#STD = -std=c89 -pedantic
#STD = -std=c99 -pedantic
STD = -std=c99
CC = gcc
MPICC = mpicc
CCFLAGS = $(OPTI) $(WARN) $(STD)

.PHONY : help move

help: Makefile
	@sed -n 's/^##//p' $<

## heat: stochastic 1d heat equation
heat_1d: heat_1d.c
	$(CC) $(CCFLAGS) -o $@.x $^ -lfftw3 -lm

clean:
	rm *.x
