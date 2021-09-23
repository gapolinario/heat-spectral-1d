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

# read path from file
file := phy_path
Phy := $(shell cat ${file})

.PHONY : help move

help: Makefile
	@sed -n 's/^##//p' $<

## move : move .c and .sh files from local to remote
move:
	scp Makefile $(Phy):/home/gbritoap/heat_1d/
	scp *.c $(Phy):/home/gbritoap/heat_1d/
	scp *.py $(Phy):/home/gbritoap/heat_1d/
	scp *.sh $(Phy):/home/gbritoap/heat_1d/

## heat: stochastic 1d heat equation
heat_1d: heat_1d.c
	$(CC) $(CCFLAGS) -o $@.x $^ -lfftw3 -lm

## heat_inplace: stochastic 1d heat equation, inplace FFTW transforms
heat_inplace_1d: heat_inplace_1d.c
	$(CC) $(CCFLAGS) -o $@.x $^ -lfftw3 -lm

## heat_jentzen: as above, with jentzen algorithm
heat_jentzen_1d: heat_jentzen_1d.c
	$(CC) $(CCFLAGS) -o $@.x $^ -lfftw3 -lm

clean:
	rm *.x
