#!/bin/bash 


# if the target file exists check to see if it is different. 
if [ -e $2 ] ; then 

    diff $1 $2 > /dev/null
    if [ $? -eq 1 ] ; then
	echo copying modified file: $1
	/bin/cp $1 $2
    fi 
else 
    echo copying non-existant file: $1
    /bin/cp $1 $2
fi


