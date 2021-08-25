'''
Author: Vijay Ravi
Date: 06/24/2021
'''

import numpy as np
import pandas as pd
import os
import csv
from pdb import set_trace as bp




def read_csv(wav2csv):

    '''
    Function to read a csv file into a dictionary. 
    Input: Path to wav2csv file.
    Output: Dictionary with wav path as key and csv path as value. 
    '''
    reader = csv.reader(open(wav2csv, 'r'))
    wav2csv_dict = {}
    for key,value in reader:
       wav2csv_dict[key] = value
    return wav2csv_dict


def read_seg_csv(csv_file):

    '''
    Funtion to read the Segmentation csv file. 
    Input: Path to the segmentation file. 
    Output: A list of lists with each item containing start time and end time of the participants segments and the total number of segments for that audio file.
    '''
    seg_time_list = []
    df = pd.read_csv(csv_file)
    df = df[df.speaker.isin(['PAR'])]
    seg_time_list = list(zip(df.begin.to_list(),df.end.to_list()))
    return seg_time_list, len(seg_time_list)


def get_segment_time(wav2csv_dict):

    '''
    Function to loop through all wav files and get segments for each from the csv file. 
    Input: Dict with wav path as key and csv path as value. 
    Output: A list of lists where each item has wav path, segment path, start time, end time and duration of the segment. 
    '''
    segInfo_list=[]
    for wav_file,csv_file in wav2csv_dict.items():
        wav_dir = os.path.dirname(wav_file)
        seg_dir = wav_dir.replace("/audio","/seg_audio")
        print("seg_dir:",seg_dir)
        seg_base = os.path.basename(wav_file)[:-4]
        print("seg_base:",seg_base)
        seg_time_list, num_seg = read_seg_csv(csv_file)
        for i in range(num_seg):
            seg_path=seg_dir+'/'+seg_base+'_seg'+str(i+1)+'.wav'
            print(seg_path)
            start,end = seg_time_list[i]
            start,end = start/1000,end/1000
            duration = end-start
            temp_list = [wav_file,seg_path,start,end,duration]
            segInfo_list.append(temp_list)
    return segInfo_list






if __name__ == "__main__":
    wav2csv = '/home/intern/summer_2021/code/extract_feats/mfcc/test/wav2csv.list' # provide the full path of the wav2csv file. 
    seg_file = '/home/intern/summer_2021/code/extract_feats/mfcc/test/seg_file.csv'
    wav2csv_dict = read_csv(wav2csv)
    segInfo_list = get_segment_time(wav2csv_dict)
  #  bp()

    # write the list into a csv file to split audio into segments using sox.
    with open(seg_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows(segInfo_list)
