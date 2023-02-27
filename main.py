import os
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
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


BATERIA = os.path.join('datas','Bateria')
GUITARRA_ACUSTICA = os.path.join('datas','Guitarra_acustica')
GUITARRA_ELECTRICA = os.path.join('datas','Guitarra_Electrica')
MARIMBA = os.path.join('datas','Marimba')
PIANO = os.path.join('datas','Piano')
VIOLIN = os.path.join('datas','Violin')

bateria = tf.data.Dataset.list_files(BATERIA+'/*.wav')
guitarra_acus = tf.data.Dataset.list_files(GUITARRA_ACUSTICA+'/*.wav')
guitarra_elec = tf.data.Dataset.list_files(GUITARRA_ELECTRICA+'/*.wav')
marimba = tf.data.Dataset.list_files(MARIMBA+'/*.wav')
piano = tf.data.Dataset.list_files(PIANO+'/*.wav')
violin = tf.data.Dataset.list_files(VIOLIN+'/*.wav')

baterias = tf.data.Dataset.zip((bateria, tf.data.Dataset.from_tensor_slices(tf.fill(len(bateria),0))))
guitarrasAcusticas = tf.data.Dataset.zip((guitarra_acus, tf.data.Dataset.from_tensor_slices(tf.fill(len(guitarra_acus),1))))
guitarrasElectricas = tf.data.Dataset.zip((guitarra_elec, tf.data.Dataset.from_tensor_slices(tf.fill(len(guitarra_elec),2))))
marimbas = tf.data.Dataset.zip((marimba, tf.data.Dataset.from_tensor_slices(tf.fill(len(marimba),3))))
pianos = tf.data.Dataset.zip((piano, tf.data.Dataset.from_tensor_slices(tf.fill(len(piano),4))))
violines = tf.data.Dataset.zip((violin, tf.data.Dataset.from_tensor_slices(tf.fill(len(violin),5))))
data = baterias.concatenate(guitarrasAcusticas).concatenate(guitarrasElectricas).concatenate(marimbas).concatenate(pianos).concatenate(violines)


for i in range(20):
    print(data.shuffle(900).as_numpy_iterator().next())

lengths = []
for file in os.listdir(os.path.join('datas','Bateria')):
    tensor_wave = load_wav_instruments(os.path.join('datas', 'Bateria', file))
    lengths.append(len(tensor_wave))
print(lengths)
print(tf.math.reduce_mean(lengths))
print(tf.math.reduce_mean(lengths)/16000)

def preprocess(file_path, label):
    wav = load_wav_instruments(file_path)
    wav = wav[:32000]
    zero_padding = tf.zeros([32000]-tf.shape(wav),dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectogram = tf.abs(spectogram)
    spectogram = tf.expand_dims(spectogram, axis=2)
    return spectogram, label

filepath, label = baterias.shuffle(buffer_size=100).as_numpy_iterator().next()
spectogram, label = preprocess(filepath,label)
plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectogram)[0])
plt.show()

filepath, label = guitarrasAcusticas.shuffle(buffer_size=100).as_numpy_iterator().next()
spectogram, label = preprocess(filepath,label)
plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectogram)[0])
plt.show()

filepath, label = guitarrasElectricas.shuffle(buffer_size=100).as_numpy_iterator().next()
spectogram, label = preprocess(filepath,label)
plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectogram)[0])
plt.show()

filepath, label = marimbas.shuffle(buffer_size=100).as_numpy_iterator().next()
spectogram, label = preprocess(filepath,label)
plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectogram)[0])
plt.show()

filepath, label = pianos.shuffle(buffer_size=100).as_numpy_iterator().next()
spectogram, label = preprocess(filepath,label)
plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectogram)[0])
plt.show()

filepath, label = violines.shuffle(buffer_size=100).as_numpy_iterator().next()
spectogram, label = preprocess(filepath,label)
plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectogram)[0])
plt.show()

print(len(data))
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)
print(len(data))
print(len(data)*.9)

train = data.take(53)
test = data.skip(53).take(6)

sample, labels = train.as_numpy_iterator().next()
print(sample.shape)

sample2, labels2 = test.as_numpy_iterator().next()
print(sample2.shape)

modeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3), activation = 'relu', input_shape=(991,257,1)),
    tf.keras.layers.Conv2D(16,(3,3), activation = 'relu', input_shape=(991,257,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dense(6, activation = 'sigmoid'),
])

modeloCNN.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics =['accuracy']
)

tensorboard_callback = TensorBoard(log_dir='logs')

hist = modeloCNN.fit(train, epochs=8, validation_data=test, callbacks=[tensorboard_callback])

# Ejecutar comando
# tensorboard --logdir logs

plt.title('Loss')
plt.plot(hist.history['loss'],'r')
plt.plot(hist.history['val_loss'],'b')
plt.show()

plt.title('Precision')
plt.plot(hist.history['precision'],'r')
plt.plot(hist.history['val_presicion'],'b')
plt.show()