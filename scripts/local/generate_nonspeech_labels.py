#! /usr/bin/python 

import sys
import os

def sum_strings(list_of_strings):
    s = 0
    for i in list_of_strings:
        s += int(i)
    return s
 

def revert_labels(in_ark, out_ark, mode):
    """ Takes vad labels of 1s and 0s and turns it to label of 0s and 1s.
        (speech labels to non-speech labels).
        We can't simply replace 1s with 0s and vice-versa. This function
        makes sure both labels maintain high false alarms (assuming the input)
        has high false alarm. 
    """
    if (mode == 's2n'):
        # converting speech to non-speech labels.
        orig = 1
        target = 0
    elif (mode == 'n2s'):
        orig = 0
        target = 1
    N = 20 # window size
    fin = open(in_ark)
    fout = open(out_ark,'w')
    for i in fin:
        uttid = i.split(' ')[0].strip()
        fout.write(uttid+' [ ')
        labels = i.split('[')[-1].split(']')[0].strip().split(' ')
        n = 0
        for j in labels:
            if (n <= N and n > (len(labels) - N)) :
                k = False
            else:
                if (sum_strings(labels[n-N:n+N])==2*N*orig): # all samples must be identical
                    k = True
                else:
                    k = False
            if k:
                fout.write(str(target)+' ')
            elif not(k):
                fout.write(str(orig)+' ')
            n+=1        
        fout.write(' ]\n')
    fin.close()
    fout.close()


if __name__=='__main__':
    os.system('. path.sh')
    scp_file = sys.argv[1]
    no_extention = scp_file[:-4]
    ark_file = no_extention+'.ark'
    os.system('copy-vector scp:'+scp_file+' ark,t:'+ark_file+'.txt') 
    revert_labels(ark_file+'.txt', ark_file+'.n.txt', 's2n')
    os.system('copy-vector ark:'+ark_file+'.n.txt'+' ark,scp:'+ark_file+'.n,'+no_extention+'.n.scp')
