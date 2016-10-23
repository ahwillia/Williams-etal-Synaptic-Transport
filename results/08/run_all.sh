#!/bin/bash
for i in $( ls 00_* ); do
    ipython $i
done
