import os


#num_trials = 5
num_splits=5
train_spk_file = '/home/intern/summer_2021/ctrl_files/diagnosis/train_all_spk'
test_spk_file = '/home/intern/summer_2021/ctrl_files/diagnosis/test_all_spk'
test_label_file = '/home/intern/summer_2021/ctrl_files/diagnosis/test_results_task1_groundtruth.csv'

#train_path='/home/intern/summer_2021/ctrl_files/seg_feat.list'
#test_path='/home/intern/summer_2021/ctrl_files/mfcc_test_path.list'

train_path='/home/intern/summer_2021/feats/eGeMAPS/train/eGeMAPS_npy_path.list'
#'/home/intern/summer_2021/feats/compare16/train/compare16_npy_path.list'
        
test_path='/home/intern/summer_2021/feats/eGeMAPS/test/test_npy_path.list'
#'/home/intern/summer_2021/feats/compare16/test/test_npy_path.list'
        
acoustic_feats = 'eGeMAPS'
#'compare16' 

task = 'diagnosis' # diagnosis or prediction of MMSE levels. 
model_name = 'svm' #svm or decision trees. 

feat_folder = '/home/intern/summer_2021/feats/'+acoustic_feats+'/'
output_path = '/home/intern/summer_2021/model_outputs'
output_folder = output_path+'/'+model_name+'_'+acoustic_feats+'/' #svm_mfcc

if not os.path.exists(output_folder):
	os.mkdir(output_folder)
