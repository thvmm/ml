#!/bin/bash 
python svm.py -t head_4k_spam_train.txt -v tail_1k_spam_train.txt -e 5
python svm.py -t head_4k_spam_train.txt -v tail_1k_spam_train.txt -e 4
python svm.py -t head_4k_spam_train.txt -v tail_1k_spam_train.txt -e 3
python svm.py -t head_4k_spam_train.txt -v tail_1k_spam_train.txt -e 2
python svm.py -t head_4k_spam_train.txt -v tail_1k_spam_train.txt -e 1
python svm.py -t head_4k_spam_train.txt -v tail_1k_spam_train.txt -e 0
python svm.py -t head_4k_spam_train.txt -v tail_1k_spam_train.txt -e -1
python svm.py -t head_4k_spam_train.txt -v tail_1k_spam_train.txt -e -2
python svm.py -t head_4k_spam_train.txt -v tail_1k_spam_train.txt -e -3
python svm.py -t head_4k_spam_train.txt -v tail_1k_spam_train.txt -e -4
python svm.py -t head_4k_spam_train.txt -v tail_1k_spam_train.txt -e -5
python svm.py -t head_4k_spam_train.txt -v tail_1k_spam_train.txt -e -6
python svm.py -t head_4k_spam_train.txt -v tail_1k_spam_train.txt -e -7
python svm.py -t head_4k_spam_train.txt -v tail_1k_spam_train.txt -e -8
python svm.py -t head_4k_spam_train.txt -v tail_1k_spam_train.txt -e -9
