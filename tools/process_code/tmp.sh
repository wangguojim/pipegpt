#!/bin/bash
dir='/raid/nlp/meta_data/zhiyuan/WuDaoCorpus2.0_base_200G'
for file in `ls $dir`
do 
  echo ${file%.*}

done
