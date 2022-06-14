# BigVGAN: A Universal Neural Vocoder with Large-Scale Training
![image](https://user-images.githubusercontent.com/56749640/173193781-0ee419a5-df66-4b94-8585-546167ecceb5.png)


In this repository, I try to implement BigVGAN (specifically BigVGAN-base model) [[Paper]](https://arxiv.org/pdf/2206.04658.pdf) [[Demo]](https://bigvgan-demo.github.io/).

## Pre-requisites
0. Pytorch >=3.9 and torchaudio >= 0.9

0. Download datasets
    1. Download the VCTK dataset
    

## Training Exmaple
```sh
# VCTK
python preprocess.py

python train_bigvgan_vocoder.py -c configs/vctk_bigvgan.json -m bigvgan
```

## 2022-06-13 (VITS with BigVGAN)
- Build Monotonic Alignment Search first ([https://github.com/jaywalnut310/vits](https://github.com/jaywalnut310/vits))
```sh
# Monotonic alignment search
Build MAS (https://github.com/jaywalnut310/vits)

python train_vits_with_bigvgan.py -c configs/vctk_bigvgan_vits.json  -m vits_with_bigvgan
```

## 2022-06-12
- Current ver has some redundant parts in some modules (e.g., data_utils have some TTS module. Ignore it plz)

## BigVGAN vs HiFi-GAN 

1. Leaky Relu --> x + (1/a)*sin^2(ax)

2. MRF --> AMP block 
- Up --> Low-pass filter --> Snake1D --> Down --> Low-pass filter

3. MSD --> MRD (Univnet discriminator) [UnivNet unofficial github](https://github.com/mindslab-ai/univnet/blob/9bb2b54838bb6d7ce767131cc7b8b61198bc7558/model/mrd.py#L49)

4. max_fre 8000 --> 12,000 for universal vocoder (sr: 24,000)
- We use the sampling rate of 22,050 and linear spectrogram for input speech.

## Low-pass filter
- [ ] pytorch extension ver with scipy??? [StarGAN3](https://github.com/NVlabs/stylegan3/blob/b1a62b91b18824cf58b533f75f660b073799595d/training/networks_stylegan3.py)
- [X] torchaudio ver (0.9 >=) can use Kaiser window and rolloff (cutoff frequency)

[Torchaudio tutorial](https://tutorials.pytorch.kr/beginner/audio_resampling_tutorial.html)

torchaudio.transforms is much faster than torchaudio.functional when resampling multiple waveforms using the same paramters. 

```sh
# beta 
m = 2
n = 6
f_h = 0.6/m
A = 2.285*((m*n)/2 - 1)*math.pi*4*f_h+7.95
beta = 0.1102*(A-8.7)
4.663800127934911
```

- (2022-06-14) Roll-off (https://pytorch.org/audio/main/_modules/torchaudio/functional/functional.html)
```sh
# In torchaudio.functional (https://pytorch.org/audio/main/_modules/torchaudio/functional/functional.html)
 base_freq = min(orig_freq, new_freq)
    # This will perform antialiasing filtering by removing the highest frequencies.
    # At first I thought I only needed this when downsampling, but when upsampling
    # you will get edge artifacts without this, as the edge is equivalent to zero padding,
    # which will add high freq artifacts.
 base_freq *= rolloff
 width = math.ceil(lowpass_filter_width * orig_freq / base_freq)

```
```
# When upsampling,  width = math.ceil(lowpass_filter_width / rolloff)
# When downsampling,  width = math.ceil(2*lowpass_filter_width / rolloff)

Hence, I think that using rolloff=0.25 may restrict the bandwith under nyquist freq (fs/2).

Somebody... check it again Please 😢
```
- (2022-06-12) I need to review this module. I used torchaudio to implement up/downsampling with low-pass filter. I used the rolloff value of 0.25 in the T.resample but I'm not sure this value is equal to sr/(2*m) where m is 2.

 torchaudio > 0.9 is needed to use the rolloff parameter for anti-aliasing in T.resample
 
 There are some issues of STFT function in pytorch of 3.9 (When using mixed precision, waveform need to be changed to float before stft ) 
 
## Results
![image](https://user-images.githubusercontent.com/56749640/173265977-f77d6e54-f723-4547-a29c-b669b43f47cb.png)

[Audio](https://github.com/sh-lee-prml/BigVGAN/tree/main/audio)

 I train the BigVGAN-base model with batch size of 64 (using two A100 GPU) and an initial learning rate of 2 × 10<sup>−4</sup>

(In original paper, BigVGAN (large model) uses batch size of 32 and an initial learning rate of 1 × 10<sup>−4</sup>
 to avoid an early training collapse)

For the BigVGAN-base model, I have not yet experienced an early training collapse with batch size of 64 and an initial learning rate of 2 × 10<sup>−4</sup>


## Automatic Mixed Precision (AMP)

The original paper may not use the AMP during training but this implementation includes AMP. Hence, the results may be different in original setting.

For training with AMP, I change the dtype of representation to float for torchaudio resampling (Need to be changed for precise transformation).

## Reference
- BigVGAN: https://arxiv.org/abs/2206.04658
- VITS: https://github.com/jaywalnut310/vits
- UnivNET: https://github.com/mindslab-ai/univnet
