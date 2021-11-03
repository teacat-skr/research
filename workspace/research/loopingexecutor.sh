#!/bin/sh

for i in {1..2}
do
    python dd.py -k $i 500000 0.1
    echo "finished width$i"
done