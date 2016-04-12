#!/bin/bash 

#python perceptron.py -f dev_file -t
python perceptron.py -t h100_train -v tail_1k_spam_train.txt
python perceptron.py -t h200_train -v tail_1k_spam_train.txt
python perceptron.py -t h400_train -v tail_1k_spam_train.txt
python perceptron.py -t h800_train -v tail_1k_spam_train.txt
python perceptron.py -t h2k_train -v tail_1k_spam_train.txt
python perceptron.py -t head_4k_spam_train.txt -v tail_1k_spam_train.txt

