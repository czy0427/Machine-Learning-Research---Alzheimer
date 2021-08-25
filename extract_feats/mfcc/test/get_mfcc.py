import os
import csv
import numpy as np

import librosa

def extract_mfcc(path_to_seg):
    print(path_to_seg)
    y,sr=librosa.load(path_to_seg)
    mfcc_feat=librosa.feature.mfcc(y=y,sr=sr)
    return mfcc_feat

def write_feat(mfcc_feat,output_path):
   # outpath=output_path
   # outfile=open(outpath,"w")
   # outfile.write(mfcc_feat)
   # outfile.close()
   np.save(output_path,mfcc_feat)

def read_csv(csv_path):
    reader=csv.reader(open(p2p,'r'))
    p2p_dict={}
    for key,value in reader:
        p2p_dict[key]=value
    return p2p_dict

if __name__ == "__main__":
    p2p='/home/intern/summer_2021/code/extract_feats/mfcc/test/p2p.csv'
    p2p_dict=read_csv(p2p)
    for wav_path,npy_path in p2p_dict.items():
       #wav_dir=os.path.dirname(wav_path)
       write_feat(extract_mfcc(wav_path),npy_path)
