#
# Copyright (c) 2014      Los Alamos National Security, LLC
#                         All rights reserved.
#

SHELL = /bin/sh

LSCI_DIRS := \
lib \
test

.PHONY: all clean install

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

install: ./lib/liblsci.a ./lib/lsci.h
ifndef LSCI_PREFIX
	$(error LSCI_PREFIX variable is not defined... Aborting build.)
endif
	mkdir -p ${LSCI_PREFIX}/lib
	install -m 0755 ./lib/liblsci.a ${LSCI_PREFIX}/lib
	mkdir -p ${LSCI_PREFIX}/include/legion
	install -m 0644 ./lib/lsci.h ${LSCI_PREFIX}/include/legion
