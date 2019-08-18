#!/bin/bash
#
# file: run.sh
# In order to reduce the workload of testing algorithms, this bash script offers an example
# to call models to generate predictions on a certain folder with data (mat files),
# by calling an example script "challenge.py", to generate "answers.csv".
# All teams should modify this file with all the command lines needed, 
# and generate predictions with exact the same layout of "answers.csv"


echo "==== running entry script on the test set ===="

# Clear previous answers.csv, if needed
rm -f answers.csv


# define a time duration calculater
function getTiming() {
    start=$1
    end=$2
    start_s=$(echo $start | cut -d '.' -f 1)
    start_ns=$(echo $start | cut -d '.' -f 2)
    end_s=$(echo $end | cut -d '.' -f 1)
    end_ns=$(echo $end | cut -d '.' -f 2)
    time=$(( ( 10#$end_s - 10#$start_s ) * 1000 + ( 10#$end_ns / 1000000 - 10#$start_ns / 1000000 ) ))
    echo "$time ms"
}

# Generate new answers.csv
start=$(date +%s.%N)

# run.sh 文件的修改部分
#python challenge.py --test_path /media/jdcloud/Val

#因为python环境用的是venv不是conda，可能需要source激活一下
source /media/uuser/data
matlab -nodesktop -nosplash -r "userpath=/media/uuser/data/final_run/; test_path=/media/jdcloud/Val/; run_matlab"
python challenge.py --test_path /media/jdcloud/Val/



end=$(date +%s.%N)
runtime=$(getTiming $start $end)
echo "runtime: "$runtime

echo "=================== Done! ===================="


