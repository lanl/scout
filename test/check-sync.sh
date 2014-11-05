#!/bin/sh
#check if sc and sc++ tests are in sync
ls -lR scc | grep "\.sc" | awk '{print $9}' | sort | sed 's/\.sc$//' > /tmp/sc.files
ls -lR sc++ | grep "\.scpp" | awk '{print $9}' |sort | sed 's/\.scpp$//' > /tmp/scpp.files 
diff /tmp/sc.files /tmp/scpp.files 
