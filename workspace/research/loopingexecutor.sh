#!/bin/sh

for i in {5..7}
do
    python dd.py -k $i 4000 0.15
    echo "finished width$i"
done