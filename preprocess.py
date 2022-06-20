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

    speakers = [os.path.basename(i) for i in glob.glob(os.path.join(args.input_path,'txt/*'))]
    speakers.remove('s5')

    for speaker in tqdm(speakers):
        os.makedirs(os.path.join(args.output_path,speaker,'train'),exist_ok=True)
        os.makedirs(os.path.join(args.output_path,speaker,'test'),exist_ok=True)

        wavs = sorted(glob.glob(os.path.join(args.input_path,'wav48_silence_trimmed',speaker,'*mic1.flac')))
        for wav in wavs[:25]:
            data = preprocess_wav(wav, hparams)
            np.savez(os.path.join(args.output_path,speaker,'test',os.path.basename(wav).replace('_mic1.flac','.npz')),
                    **data, allow_pickle=False)
        
        for wav in wavs[25:]:
            data = preprocess_wav(wav, hparams)
            np.savez(os.path.join(args.output_path,speaker,'train',os.path.basename(wav).replace('_mic1.flac','.npz')),
                    **data, allow_pickle=False)


def preprocess_wav(wav, hparams):
    
    audio, _ = librosa.load(wav, sr=22050)
    audio = audio * hparams.data.max_wav_value
    
    text_file = wav.replace('wav48_silence_trimmed','txt').replace('_mic1.flac','.txt')
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
