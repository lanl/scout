#
# Copyright (c) 2014, Los Alamos National Security, LLC All rights reserved.
#

ALL_TESTS = \
c-veccp

all: c-veccp

CFLAGS = \
-Wall -Wextra \
-I../lib \
-I$(LG_RT_DIR)

LDFLAGS = \
-L../lib -llsci \
-lstdc++

clean:
	rm -f $(ALL_TESTS)
