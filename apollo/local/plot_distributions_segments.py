#! /usr/bin/python 

import sys
import pylab 
import numpy as np

def read_scores(fin):
    out_list = []
    for i in fin:
        line_list = i.strip().split(' ')
        score = float(line_list[-1])
        out_list.append(score)
    return out_list

ami_scores = sys.argv[1]
apollo_scores = sys.argv[2]

fami = open(ami_scores)
ami = read_scores(fami)
fami.close()

fapollo = open(apollo_scores)
apollo = read_scores(fapollo)
fapollo.close()

x1 = np.array(ami)
x1 = x1.reshape((len(ami),1))
x2 = np.array(apollo)
x2 = x2.reshape((len(apollo),1))
label1 = ['AMI']
pylab.hist(x1, 500, normed=1, cumulative=1, color=['red'], fc='none', lw=1.5, histtype='step', label=label1, alpha=1)

label2 = ['Apollo']
pylab.hist(x2, 300, normed=1, cumulative=1, color=['green'], fc='none', lw=1.5, histtype='step', label=label2, alpha=1)
pylab.legend(prop={'size': 20}, loc = 4)
pylab.xlim([0,10])
pylab.ylim([0,1])
pylab.xlabel('speech segment (seconds)',fontsize=18)
pylab.ylabel('normalized cumulative sum', fontsize=18)
pylab.xticks(np.linspace(0,10,11),fontsize = 15)
pylab.yticks(np.linspace(0,1,11),fontsize = 15)
pylab.show()
