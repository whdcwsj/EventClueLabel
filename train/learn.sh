#!/usr/bin/env bash

# https://developer.aliyun.com/article/638420?spm=a2c6h.12873639.article-detail.23.301f42abwMrjmh

#1、变量定义
#MY_NAME="王思杰"
#echo "Hello,I am ${MY_NAME}"

#2、键盘的输入
#read -p "enter your name: " NAME
#echo "Your name is: ${NAME}"

#3、if-else判断
#m=1
#n=2
#
#if [ $n -eq $m ]
#then
#        echo "Both variables are the same"
#else
#        echo "Both variables are different"
#fi
#
#n=12
#if [ $((n%2)) = 0 ]
#then
#  echo "The number is even."
#else
#  echo "The number is odd."
#fi


#4、case语句
#read -p "Enter the answer in Y/N: " ANSWER
#case "$ANSWER" in
#  [yY] | [yY][eE][sS])
#    echo "The Answer is Yes :)"
#    ;;
#  [nN] | [nN][oO])
#    echo "The Answer is No :("
#    ;;
#  *)
#    echo "Invalid Answer :/"
#    ;;
#esac


#5、for循环
#COLORS="red green blue"
#for COLOR in $COLORS
#do
#  echo "The Color is: ${COLOR}"
#done



#6、参数传递
#for example:
#./script.sh param1 param2 param3 param4
#$0 -- "script.sh"
#$1 -- "param1"
#$2 -- "param2"
#$3 -- "param3"
#$4 -- "param4"
#$@ -- array of all positional parameters 存储所有参数


#上一条命令执行后的退出状态码被保存在变量$?中
#HOST="google.com"
#ping -c 1 $HOST     # -c is used for count, it will send the request, number of times mentioned
#RETURN_CODE=$?
#if [ "$RETURN_CODE" -eq "0" ]
#then
#  echo "$HOST reachable"
#else
#  echo "$HOST unreachable"
#fi


#7、调用函数
#function myFunc() {
#    echo "Shell Scripting Is Fun!"
#}
#myFunc




python trainer.py --flag_id 0 &  python trainer.py --flag_id 1 & python trainer.py --flag_id 2






#test_num=${RANDOM};
#echo "Test start. Current process is: $$. Parent process is: ${PPID}. Test_num is: ${test_num}. ";
## &
#{
#echo '-----------&------------';
#echo "& test start. test_num is: ${test_num} ";
#sleep 30
#echo "& test. Now pid is:$$";
#test_num=${RANDOM}
#echo "& test_num is: ${test_num}. ";
#}&
#echo "& test end. Test_num is: ${test_num}. ";