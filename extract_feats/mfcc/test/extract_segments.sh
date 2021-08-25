#!/bin/bash
source ~/.bashrc

# Author: Vijay Ravi
# Date: 06/24/2021


while read segName; do
  wav_file=$(echo $segName | cut -f1 -d,)
  seg_file=$(echo $segName | cut -f2 -d,)
  start_time=$(echo $segName | cut -f3 -d,)
  duration=$(echo $segName | cut -f5 -d,| bc)
  sox $wav_file $seg_file trim $start_time $duration
done < seg_file.csv
