#!/bin/sh

for i in {11..13}
do
    python dd.py -k $i 4000 0.15
    echo "finished width$i"
done