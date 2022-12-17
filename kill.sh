#!/bin/bash
# fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' |sudo sh
# sudo systemctl restart nvidia-fabricmanager
ps aux|grep megatron |awk   '{print $2}'|xargs kill -9
#killall python
