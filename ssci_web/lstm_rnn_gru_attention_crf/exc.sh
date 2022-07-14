#!bin/sh/
nohup python train.py data/train/train0 data/test/test0 > log0.txt &
sleep 10s
nohup python train.py data/train/train1 data/test/test1 > log1.txt &
sleep 10s
nohup python train.py data/train/train2 data/test/test2 > log2.txt &
sleep 10s
nohup python train.py data/train/train3 data/test/test3 > log3.txt &
sleep 10s
nohup python train.py data/train/train3 data/test/test4 > log4.txt &

