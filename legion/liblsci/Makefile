#
# Copyright (c) 2014      Los Alamos National Security, LLC
#                         All rights reserved.
#

SHELL = /bin/sh

LSCI_DIRS := \
lib \
test

.PHONY: all clean

all:
	@ \
	for i in ${LSCI_DIRS}; do \
	    echo "making: $$i ..."; \
	    (cd $$i && make); \
	    echo ""; \
	done

clean:
	@ \
	for i in ${LSCI_DIRS}; do \
	    echo "cleaning: $$i ..."; \
	    (cd $$i && make clean); \
	    echo ""; \
	done
