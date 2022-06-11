import utils
import argparse
import json
import glob
import os
import numpy as np
import librosa
from tqdm import tqdm
import scipy.signal as sps
from text import text_to_sequence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/vctk.json",
                      help='JSON file for configuration')
    parser.add_argument('-i', '--input_path', type=str, default="./dataset/VCTK-Corpus")
    parser.add_argument('-o', '--output_path', type=str, default="./dataset/VCTK-Corpus/preprocessed_npz")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = utils.HParams(**config)

    speakers = [os.path.basename(i) for i in glob.glob(os.path.join(args.input_path,'wav48/*'))]

    for speaker in tqdm(speakers):
        os.makedirs(os.path.join(args.output_path,speaker,'train'),exist_ok=True)
        os.makedirs(os.path.join(args.output_path,speaker,'test'),exist_ok=True)

        wavs = sorted(glob.glob(os.path.join(args.input_path,'wav48',speaker,'*.wav')))
        for wav in wavs[:25]:
            data = preprocess_wav(wav, hparams)
            np.savez(os.path.join(args.output_path,speaker,'test',os.path.basename(wav).replace('.wav','.npz')),
                    **data, allow_pickle=False)
        
        for wav in wavs[25:]:
            data = preprocess_wav(wav, hparams)
            np.savez(os.path.join(args.output_path,speaker,'train',os.path.basename(wav).replace('.wav','.npz')),
                    **data, allow_pickle=False)

def downsample(audio, sr):
    num = round(len(audio)*float(22050) / float(sr))
    return sps.resample(audio, num)

def preprocess_wav(wav, hparams):
    audio, sr = utils.load_wav_to_torch(wav)
    audio, _ = librosa.effects.trim(np.array(audio), 
                                    top_db=hparams.data.top_db,
                                    frame_length=hparams.data.filter_length,
                                    hop_length=hparams.data.hop_length)

    if sr != hparams.data.sampling_rate:
        audio = downsample(audio, sr)
    
    text_file = wav.replace('wav48','txt').replace('.wav','.txt')
    with open(text_file, encoding='utf8') as f:
        text = f.readline().rstrip()
    token = text_to_sequence(text, ["english_cleaners2"]) 

    data = {
        'audio': audio,
        'token': token,
        'text': text
    }

    return data




if __name__ == "__main__":
    main()