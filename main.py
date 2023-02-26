import os
from matplotlib import pyplot as plt
import tensorboard
import tensorflow as tf
import tensorflow_io as tfio


BATERIA_AUDIO = os.path.join('datas','Bateria','bateria_001.wav')
GUITARRA_ACUSTICA_AUDIO = os.path.join('datas','Guitarra_acustica','Guitarra_actustica_001.wav')
GUITARRA_ELECTRICA_AUDIO = os.path.join('datas','Guitarra_Electrica','guitarra_electrica_001.wav')
MARIMBA_AUDIO = os.path.join('datas','Marimba','Marimbaa_001.wav')
PIANO_AUDIO = os.path.join('datas','Piano','piano_01.wav')
VIOLIN_AUDIO = os.path.join('datas','Violin','Violin_002.wav')


def load_wav_instruments(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

batWave = load_wav_instruments(BATERIA_AUDIO)
geWave = load_wav_instruments(GUITARRA_ELECTRICA_AUDIO)
gatWave = load_wav_instruments(GUITARRA_ACUSTICA_AUDIO)
marWave = load_wav_instruments(MARIMBA_AUDIO)
pianoWave = load_wav_instruments(PIANO_AUDIO)
viWave = load_wav_instruments(VIOLIN_AUDIO)

plt.plot(batWave)
plt.plot(geWave)
plt.plot(gatWave)
plt.plot(marWave)
plt.plot(pianoWave)
plt.plot(viWave)
plt.show()