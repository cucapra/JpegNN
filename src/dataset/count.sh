#!/bin/bash
j=0
for i in $(seq 1 90)
do
    num=$(ls /data/zhijing/coco/train/$i | wc -l)
    if [ $num -gt 300 ]
    then
        echo ${i}, $num
        j=`expr ${j} + 1`
    fi
done
echo $j
