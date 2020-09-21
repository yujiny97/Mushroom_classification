#!/bin/bash

for x in `find ./train/ -maxdepth 2 -mindepth 1 -type d -print`
do
echo $x, `find $x -type f|wc -l`;
done
