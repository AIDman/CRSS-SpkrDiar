#! /usr/bin/python 
import numpy as np
from scipy.io import wavfile
import sys
import os
import pylab
import pdb

def deframe(x_frames, winlen, hoplen):
    '''
    interpolates 1-dimensional framed data into persample values. 
    '''
    n_frames = len(x_frames)
    n_samples = n_frames*hoplen + winlen
    x_samples = np.zeros((n_samples,1))
    for ii in range(n_frames):
        x_samples[ii*hoplen : ii*hoplen + winlen] = x_frames[ii]
    return x_samples

def read_ark_vad(vad_fn):
    vad_dict = {}
    fvad = open(vad_fn)
    for i in fvad:
        vadid = i.split('[')[0].strip()
        vad_scores = i.split('[')[1].split(']')[0].strip()
        vad_scores_list = []
        for j in vad_scores.split(' '):
            if (j!=''):
                vad_scores_list.append(float(j.strip()))
        vad_scores_array = np.array(vad_scores_list)
        vad_dict[vadid] = vad_scores_array
    return vad_dict

def read_int_ark_vad(vad_fn):
    vad_dict = {}
    fvad = open(vad_fn)
    for i in fvad:
        vadid = i.split(' ')[0].strip()
        vad_scores = i.split(' ')[1:]
        vad_scores_list = []
        for j in vad_scores:
            if (j.strip()!=''):
                vad_scores_list.append(float(j))
        vad_scores_array = np.array(vad_scores_list)
        vad_dict[vadid] = vad_scores_array
    return vad_dict            

def plot_vad(wav_fn, vad_fn, winlen, hoplen, mode):
    '''
    Plot VAD labels alongside signal for comparison. This tool helps to run 
    a sanity check on the VAD outputs. This code only works on the index style
    vad labels. 
    '''
    if mode == 'ark':
        """ Read wav.scp and corresponding scores. 
            This script plots all VAD scores for files in wav.scp.
            wav_fn: wav.scp file
            vad_fn: VAD scores for each utterance in ark,t format
            """
        vad_files = read_int_ark_vad(vad_fn)
        #vad_files = read_ark_vad(vad_fn)
        fwav = open(wav_fn)
        wavs = {}
        for i in fwav:
            i = i.strip()
            uttid = i.split(' ')[0]
            uttfile = i.split(' ')[1].strip()
            wavs[uttid] = uttfile
            vad_samples = deframe(vad_files[uttid],winlen,hoplen)
            fs, s = wavfile.read(wavs[uttid])
            N1 = 5000000
            #N2 = N1 + len(vad_samples)
            N2 = N1 + 5000000
            s = s[N1:N2]
            pylab.plot(s/float(max(abs(s))))
            pylab.plot(vad_samples[N1:N2]/float(max(abs(vad_samples[N1:N2])))+0.01,'r')
            pylab.show()
            break
        return 0

if __name__ == '__main__':
    wav_fn = sys.argv[1]
    vad_fn = sys.argv[2]
    fs = int(sys.argv[3])
    plot_vad(wav_fn, vad_fn, fs*0.025, fs*0.01, 'ark')
