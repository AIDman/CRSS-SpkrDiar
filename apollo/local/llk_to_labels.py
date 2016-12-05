#! /usr/bin/python 

# Converts log-likelihood ratios to labels by thresholding the 
# scores to obtain hard decisions. We also would like to limit
# the number of hops from speech to non-speech and vice versa
# in the labels (i.e., smooth responses).
# 
import numpy as np
import sys
import os
import pylab as plt

def read_ark(in_ark):
    """
        Read ark file containing log-likelihood ratios from GMM VAD.
        """
    scores = {}
    fin = open(in_ark)
    for i in fin:
        uttid = i.split(' ')[0].strip()
        uttllks_str = i.split('[')[-1].strip().split(']')[0].strip()
        uttllks_list = uttllks_str.split(' ')
        uttllks = [float(llk) for llk in uttllks_list]
        del uttllks_list, uttllks_str
        scores[uttid] = np.array(uttllks)
    return scores

def hard_decisioning(llks):
    """ 
        Extract hard speech/non-speech decisions from log-likelihood
        ratios.
        Implement Otsu's method to minimize intra-class and maximize 
        inter-class variances, which according to Otsu are synonymous.
        """
    max_between_var = 0.
    nBins = 100
    hist, edges = np.histogram(llks,bins=nBins)
    N = len(llks)
    sumN = 0.
    wN = 0.
    muN = 0.
    thr = 0.
    sum1 = np.dot(edges[:-1],hist)
    for i in range(nBins):
        wN += hist[i]
        wS = N - wN
        if wS==0:
            break
        sumN += .5*(edges[i]+edges[i+1])*hist[i]
        muN += sumN/wN
        muS = (sum1 - sumN)/wS
        between_var = wN*wS*((muN - muS)**2)
        if (between_var > max_between_var):
            thr = edges[i]
            max_between_var = between_var
    # Now that we have the threshold, 
    labels = llks[:]
    labels[llks>thr] = 1
    labels[llks<thr] = 0
    return labels
    

def smooth_decisions(decisions):
    new_decisions = decisions[:]
    N = 100 # size of smoothing window
    delta = 20
    for n in range(N,len(decisions)-N,N):
        if (sum(decisions[n-N:n+N]) > delta):
            new_decisions[n-N:n+N] = 1
        else:
            new_decisions[n-N:n+N] = 0
    return new_decisions

if __name__=='__main__':
    score_ark = sys.argv[1]        
    out_string = sys.argv[2] 
    
    out_types = out_string.split(':')[0]
    out_paths = out_string.split(':')[-1]
    if ',' in out_types:
        if ('ark' in out_types.split(',')[0]):
            out_ark = out_paths.split(',')[0]
            out_scp = out_paths.split(',')[1]
        else:
            out_ark = out_paths.split(',')[1]
            out_scp = out_paths.split(',')[0]
    else:
        out_ark = out_types
        out_scp = out_types+'.scp'
    
    llk_dict = read_ark(score_ark)
    fout = open(out_ark,'w')
    for i in llk_dict:
        labels_i_h = hard_decisioning(llk_dict[i])
        labels_i = smooth_decisions(labels_i_h)
        fout.write(i+' [ ')
        for j in labels_i:
            lab = str(int(j))
            fout.write(lab+' ')
        fout.write(' ]\n')
    fout.close()
    #os.system('copy-vector ark:'+out_ark+' scp:'+out_scp)

            
