# Dependancies for laughter detection script.
#	1. pip install progressbar
#	2. pip install librosa
#	3. pip install keras
#	4. pip install --upgrade tensorflow




'''
Research notes on the project
	Butterworth filter:
	- A butterworth filter is a lowpass filter such that 
	- the sensitivity for the passband is constant, and rolls off
	- towards zero in the stopband.
	- The order of the filter determines the gradient of the 
	- bode plot slope after the break ferquency.
	- 1 is a lower gradient vs. 5 is a steep gradient.

'''



import os.path
import argparse                   
import sys
import os

import librosa		# Python package for music and audio analysis
import numpy as np
from keras.models import load_model	# Used to load the audio model.
import scipy.signal as signal		# Used to apply the lowpass filter.


# The sampling rate of the audio.
AUDIO_SAMPLE_RATE = 16000



# Helper funtions for the features list.

# Calculates the Mel Frequency Cepstral Coefficients.
def compute_mfcc_features(y,sr):
	
	# Gets the Mel-frequency cepstral coefficients.
	# Params are the audio sample, sampling rate, number of mfcc to return, pre-computed log powered mel
	# Spectrogram.
    mfcc_feat = librosa.feature.mfcc(y,sr,n_mfcc=12,n_mels=12,hop_length=int(sr/100), n_fft=int(sr/40)).T

    # Seperates a complex valued Spectrogram D into its magnitude and phase components.
    S, phase = librosa.magphase(librosa.stft(y,hop_length=int(sr/100)))

    # Calculates the root mean square energy for each frame.
    rms = librosa.feature.rmse(S=S).T
    return np.hstack([mfcc_feat,rms])

# Generates the local estimate of the derivative of the input data slong the selected axis.
def compute_delta_features(mfcc_feat):
    return np.vstack([librosa.feature.delta(mfcc_feat.T),librosa.feature.delta(mfcc_feat.T, order=2)]).T

def format_features(mfcc_feat, delta_feat,index, window_size=37):
    return np.append(mfcc_feat[index-window_size:index+window_size],delta_feat[index-window_size:index+window_size])



# Getting different audio features for analysis
def get_feature_list(y,sr,window_size=37):

	# Calculates the Mel Frequency Cepstral Coefficients.
	mfcc_feat = compute_mfcc_features(y,sr)

	delta_feat = compute_delta_features(mfcc_feat)
	zero_pad_mfcc = np.zeros((window_size,mfcc_feat.shape[1]))
	zero_pad_delta = np.zeros((window_size,delta_feat.shape[1]))
	padded_mfcc_feat = np.vstack([zero_pad_mfcc,mfcc_feat,zero_pad_mfcc])
	padded_delta_feat = np.vstack([zero_pad_delta,delta_feat,zero_pad_delta])
	feature_list = []
	for i in range(window_size, len(mfcc_feat) + window_size):
		feature_list.append(format_features(padded_mfcc_feat, padded_delta_feat, i, window_size))
	feature_list = np.array(feature_list)
	return feature_list

# Applying a lowpass filter to the audio.
def lowpass(sig, filter_order = 2, cutoff = 0.01):
	#Set up Butterworth filter

	filter_order  = 2

	# Create a butterworth filter of the second order with 
	# ba (numerator/denominator) output.
	B, A = signal.butter(filter_order, cutoff, output='ba')

	# Applies the linear filter twice to the signal, 
	# Once forwards, and once backwards.
	return(signal.filtfilt(B,A, sig))

# Extracts laughter from the filtered audio.
def frame_span_to_time_span(frame_span):
    return (frame_span[0] / 100., frame_span[1] / 100.)

def collapse_to_start_and_end_frame(instance_list):
    return (instance_list[0], instance_list[-1])

def get_laughter_instances(probs, threshold = 0.5, min_length = 0.2):
	instances = []
	current_list = []
	for i in xrange(len(probs)):
		if np.min(probs[i:i+1]) > threshold:
			current_list.append(i)
		else:
			if len(current_list) > 0:
				instances.append(current_list)
				current_list = []
	instances = [frame_span_to_time_span(collapse_to_start_and_end_frame(i)) for i in instances if len(i) > min_length]
	return instances


def segment_laugh(input_path, model_path, output_path, threshold, min_length):
	print('Processing audio file...')
	# Extract the audio file as a floating point time series with the 
	# returned sample rate.
	y, sr = librosa.load(input_path, sr = AUDIO_SAMPLE_RATE)
	print('Analyzing laughter...')

	# Loading the model trained to detect laughter.
	model = load_model(model_path)

	# Getting a list of different audio features for analysis.
	feature_list = get_feature_list(y,sr)


	probs = model.predict_proba(feature_list)

	# Reshaping the tensor to the specified shape.
	probs = probs.reshape((len(probs)))

	# Filtering the input signal using the butterworth filter.
	filtered = lowpass(probs)
	instances = get_laughter_instances(filtered, threshold, min_length)
	return instances


# Run this script directly to extract laughter from audio.
if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description = ('Client to detect laughter from audio files'))
	parser.add_argument('-files', action = 'store', dest = 'in_files', 
		default = None, required = True, help = "File to extract audio"\
		" from")
	parser.add_argument('-output', action = 'store', dest = 'out_dir',
		help = "Output file path", required = True, default = None)
	parser.add_argument('-model', action = 'store', dest = 'mod',
		default = './model.h5', required = False)
	parser.add_argument('-threshold', action = 'store',dest = 'thresh',
		default = 0.5, required = False)
	parser.add_argument('-min_length', action = 'store', dest = 'len',
		default = 0.2, required = False)

	args = parser.parse_args()
	instances = segment_laugh(args.in_files,args.mod, args.out_dir, args.thresh,0.2)
	for instance in instances:
		print(instance)




