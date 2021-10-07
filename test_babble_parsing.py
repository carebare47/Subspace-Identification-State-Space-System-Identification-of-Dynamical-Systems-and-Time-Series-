#!/usr/bin/python

import re
import numpy as np
file1 = open('babble_results', 'r')
Lines = file1.readlines()
train_len = len(Lines)/2.0
y = []
x = []

X_ident = []
Y_ident = []

X_val = []
Y_val = []


count = 0
# Strips the newline character

for line in Lines:
    line = line.strip()
    j=0
    x_ = ()
    y_ = []
    print("")
    print("line: {}".format(line))
    if True:
        for chunk in line.split('th'):
            if chunk == '':
                continue
            print("{}".format(chunk))
            # chunk = chunk.replace('j1@', '')
            chunk = re.sub(r'j.@', '', chunk)
            print("{}".format(chunk.split(':')))
            y_.append(float(chunk.split(':')[0]))
            print("chunk{}: {} ##### {}".format(j, chunk, chunk.split(',')))
            #x_ = tuple(float(chunk.split(':')[1].split(',')))
            x_ = x_ + tuple(float(x) for x in chunk.split(':')[1].split(','))
            print("y: {} x: {}".format(y_, x_))
            j += 1
            #break
        if count < train_len:
            X_ident.append(x_)
            Y_ident.append(tuple(y_))
        else:
            X_val.append(x_)
            Y_val.append(tuple(y_))
        count += 1
        if count == (len(Lines) - 1):
            break
            
            
            
X_ident = np.array(X_ident)
Y_ident = np.array(Y_ident)
X_val = np.array(X_val)
Y_val = np.array(Y_val)

input_ident = np.random.rand(3,int(train_len))
input_val = np.random.rand(3,int(train_len))

print("X_ident.shape {}".format(X_ident.shape))
print("Y_ident.shape {}".format(Y_ident.shape))
print("X_val.shape {}".format(X_val.shape))
print("Y_val.shape {}".format(Y_val.shape))
print("input_ident.shape {}".format(input_ident.shape))
print("input_val.shape {}".format(input_val.shape))
