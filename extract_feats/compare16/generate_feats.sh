source ~/.bashrc

#config file paths
compare16=/home/intern/summer_2021/tools/opensmile/config/compare16/ComParE_2016.conf
eGeMAPS=/home/intern/summer_2021/tools/opensmile/config/egemaps/v01b/eGeMAPSv01b.conf

while read line; do
	wav_file=$(echo $line | cut -f1 -d,)
	echo $wav_file
	output_file=$(echo $line | cut -f2 -d,)
	echo $output_file
	SMILExtract -C $compare16 -I $wav_file -D $output_file 
	#SMILExtract -C $eGeMAPS -I $wav_file -D $output_file 
#done < feats_compare16.scp
done < p2p.list

