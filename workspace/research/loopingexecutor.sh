#!/bin/sh

for i in 
do
    echo "width$i start"
    python dd.py -k $i 4000 0.15
    echo "width$i finished"
done