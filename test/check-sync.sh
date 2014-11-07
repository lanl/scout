#!/bin/sh
#check if sc and sc++ tests are in sync

#make list of scc test excluding a few
find scc -name "*.sc" \
 	| grep -v "^scc/error/mesh-param-byvalue.sc$" \
 	| grep -v "^scc/error/mesh-param-missing-star.sc$" \
        | grep -v "^scc/warning/mesh-param-builtin.sc$" \
	| sort | sed 's/\.sc$//' | sed 's/^scc\///' > /tmp/sc.files

# make list of sc++ test excluding a few
find sc++ -name "*.scpp" \
	| grep -v "^sc++/keywords.scpp$" \
 	| grep -v "^sc++/error/mesh-param-ptr.scpp$" \
	| sort | sed 's/\.scpp$//' | sed 's/^sc++\///'  > /tmp/scpp.files 
diff /tmp/sc.files /tmp/scpp.files 
