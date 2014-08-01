#
# Copyright (c) 2014, Los Alamos National Security, LLC All rights reserved.
#

ALL_TESTS = \
c-veccp \
c-unimesh \
mesh-gen-test

all: $(ALL_TESTS)

CFLAGS = \
-std=c99 \
-Wall -Wextra \
-I../lib \
-I$(LG_RT_DIR) \
-L../lib

LDFLAGS = -lstdc++ -lpthread -llsci
UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
 LDFLAGS += -lrt
endif

clean:
	rm -f $(ALL_TESTS)

.c:
	$(CC) $(CFLAGS) $@.c -o $@ $(LDFLAGS)
