import glob
import os
import pickle

import cv2
import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

import mel_features
import vggish_params

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
AUTOTUNE = tf.data.experimental.AUTOTUNE


def make_video_data(video_config, batch_size, norm=False,
                    x_start=0, x_end=600, y_start=310,
                    y_end=900, test=False):
    n_frames = video_config['n_frames']
    resolution = video_config['resolution']
    speech_filename = video_config['speech_filename']

    paths = glob.glob(speech_filename + '\*')
    speech_names = np.array([path.split('\\')[-1].strip('.mp4').split('-') for path in paths])

    frame_size = (resolution, resolution)
    imgs_video = []

    if test:
        paths = paths[:120]
        speech_names = speech_names[:120]

    for path in paths:
        cap = cv2.VideoCapture(path)
        i = 0
        imgs = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            if i % int(cap.get(7) / n_frames) == 0:
                img = cv2.resize(frame[x_start:x_end, y_start:y_end, :], frame_size)
                if norm:
                    img = img / 255.0
                imgs.append(img)
            i += 1
        cap.release()
        imgs_video.append(imgs[:n_frames])
    imgs_video = np.array(imgs_video)
    labels = speech_names[:, 2].astype(int) - 1

    imgs = imgs_video.reshape([-1, *imgs_video.shape[-3:]])
    imgs = preprocess_input(imgs)
    imgs = imgs.reshape((-1, n_frames, resolution, resolution, 3))

    X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2, random_state=42)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).cache()
    train_dataset = train_dataset.shuffle(len(y_train)).batch(batch_size)
    train_dataset = train_dataset.prefetch(AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).cache()
    val_dataset = val_dataset.cache().batch(batch_size).prefetch(AUTOTUNE)

    return train_dataset, val_dataset


def make_audio_data(audio_config, batch_size, test=False):
    path = audio_config['speech_filename']
    res1 = audio_config['res1']
    res2 = audio_config['res2']

    path_list = os.listdir(path)
    path_list.sort()
    if test:
        path_list = path_list[:120]
    mel_feature = np.zeros((len(path_list), 5, 1, res1, res2))
    label = np.zeros(len(path_list))

    for k, filename in enumerate(path_list):
        each = wavfile_to_examples(os.path.join(path, filename))
        mel_feature[k, ...] = each.detach().numpy()
        label[k] = int(filename[6:8])

    mel_feature = np.squeeze(mel_feature)
    label = label.astype(int) - 1
    X_train, X_test, y_train, y_test = train_test_split(mel_feature, label, test_size=0.2, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).cache()

    train_dataset = train_dataset.shuffle(len(y_train)).batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).cache()
    val_dataset = val_dataset.cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset


def waveform_to_examples(data, sample_rate, return_tensor=True):
    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != vggish_params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=vggish_params.SAMPLE_RATE,
        log_offset=vggish_params.LOG_OFFSET,
        window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vggish_params.NUM_MEL_BINS,
        lower_edge_hertz=vggish_params.MEL_MIN_HZ,
        upper_edge_hertz=vggish_params.MEL_MAX_HZ)

    # Frame features into examples.
    features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(
        vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(
        vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = mel_features.frame(
        log_mel,
        window_length=example_window_length,
        hop_length=example_hop_length)

    if return_tensor:
        log_mel_examples = torch.tensor(
            log_mel_examples, requires_grad=True)[:, None, :, :].float()

    return log_mel_examples


def wavfile_to_examples(wav_file, return_tensor=True):
    wav_data, sr = sf.read(wav_file, dtype='int16')
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0
    return waveform_to_examples(samples, sr, return_tensor)


def load_audio_and_video_data(batch_size, test=False, audio_file='outputdata', video_file='Video_Speech'):
    names = [name.split('.')[0][3:] for name in os.listdir(video_file)]
    if test:
        names = names[:100]

    video_paths = [os.path.join(video_file, '02-' + name + '.mp4') for name in names]
    audio_paths = [os.path.join(audio_file, '03-' + name + '.wav') for name in names]
    labels = np.array([name.split('-')[1] for name in names]).astype(int) - 1

    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        np.transpose([video_paths, audio_paths]), labels, test_size=0.2, random_state=42)
    X_train_paths = np.transpose(X_train_paths)
    X_test_paths = np.transpose(X_test_paths)

    X_train_videos, X_train_audios = load_and_preprocess_from_path(X_train_paths)
    X_test_videos, X_test_audios = load_and_preprocess_from_path(X_test_paths)

    train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_videos, X_train_audios), y_train)).cache()
    train_dataset = train_dataset.shuffle(len(y_train)).batch(batch_size)
    train_dataset = train_dataset.prefetch(AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices(((X_test_videos, X_test_audios), y_test)).cache()
    val_dataset = val_dataset.cache().batch(batch_size).prefetch(AUTOTUNE)

    return train_dataset, val_dataset


def load_and_preprocess_from_path(paths):
    video_paths, audio_paths = paths
    video_config = {
        'n_frames': 5,
        'resolution': 64,
        'speech_filename': 'Video_Speech',
        'num_channel': 3
    }
    audio_config = {
        'speech_filename': 'outputdata',
        'res1': 96,
        'res2': 64,
        'num_channel': 1
    }
    videos = load_and_preprocess_video(video_paths, video_config)
    audios = load_and_preprocess_audio(audio_paths, audio_config)
    return videos, audios


def load_and_preprocess_video(paths, video_config, x_start=0, x_end=600, y_start=310, y_end=900):
    res = video_config['resolution']
    frame_size = (res, res)
    n_frames = video_config['n_frames']
    num_channel = video_config['num_channel']

    imgs_video = []

    for path in paths:
        cap = cv2.VideoCapture(path)
        i = 0
        imgs = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            if i % int(cap.get(7) / n_frames) == 0:
                img = cv2.resize(frame[x_start:x_end, y_start:y_end, :], frame_size)
                imgs.append(img)
            i += 1
        cap.release()
        imgs_video.append(imgs[:n_frames])
    imgs_video = np.array(imgs_video)

    imgs = imgs_video.reshape([-1, *imgs_video.shape[-3:]])
    imgs = preprocess_input(imgs)
    imgs = imgs.reshape((-1, n_frames, res, res, num_channel))

    return imgs


def load_and_preprocess_audio(paths, audio_config):
    res1 = audio_config['res1']
    res2 = audio_config['res2']
    mel_feature = np.zeros((len(paths), 5, 1, res1, res2))

    for k, path in enumerate(paths):
        each = wavfile_to_examples(path)
        mel_feature[k, ...] = each.detach().numpy()

    return np.squeeze(mel_feature)


class MultiModalModel(tf.keras.Model):
    def __init__(self, video_config, audio_config, num_label=None,
                 seq_length=None, d_embedding=None, vgg_cfg=None,
                 video_model_path=None, audio_model_path=None):
        super(MultiModalModel, self).__init__()

        self.audio_config = audio_config
        self.video_config = video_config

        if video_model_path is None:
            self.video_encoder = VideoEncoder(
                num_label=num_label, seq_length=seq_length, d_embedding=d_embedding)
        else:
            self.video_encoder = tf.keras.models.load_model(video_model_path, compile=False)

        if audio_model_path is None:
            self.audio_encoder = AudioEncoder(
                vgg_cfg=vgg_cfg, num_label=num_label, d_embedding=d_embedding)
        else:
            self.audio_encoder = tf.keras.models.load_model(audio_model_path, compile=False)

        self.fusion_model = FusionNetwork(seq_length=seq_length, d_embedding=d_embedding)
        self.classifier = DenseClassifier(num_label=num_label)

        self.optimizer = tf.keras.optimizers.Adam()
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.train_loss = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_loss = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)

    def train_full_model(self, EPOCHS, batch_size, video_encoder_training, audio_encoder_training, test=False):

        if not self.audio_encoder.trained:
            self.train_audio_encoder(EPOCHS=EPOCHS, batch_size=batch_size, test=test)

        if not self.video_encoder.trained:
            self.train_video_encoder(EPOCHS=EPOCHS, batch_size=batch_size, test=test)

        train_dataset, val_dataset = load_audio_and_video_data(batch_size, test=test)

        train_log = {
            'train': {'loss': [], 'accuracy': []},
            'val': {'loss': [], 'accuracy': []}
        }

        for epoch in range(EPOCHS):

            self.train_loss.reset_states()
            self.train_acc.reset_states()
            self.val_loss.reset_states()
            self.val_acc.reset_states()

            for (batch, ((v, a), target)) in enumerate(train_dataset):
                self._train_full_model_step(v, a, target, video_encoder_training, audio_encoder_training)

                train_log['train']['loss'].append(self.train_loss.result())
                train_log['train']['accuracy'].append(self.train_acc.result())

                if batch % 5 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result(), self.train_acc.result()
                    ))
            self.evaluate_full_model(val_dataset, epoch)

            train_log['val']['loss'].append(self.val_loss.result())
            train_log['val']['accuracy'].append(self.val_acc.result())

        with open('finetune_full_model_train_history.pickle', 'wb') as f:
            pickle.dump(train_log, f)

    def _train_full_model_step(self, v, a, target, video_encoder_training, audio_encoder_training):

        au_res1 = self.audio_config['res1']
        au_res2 = self.audio_config['res2']
        au_num_channel = self.audio_config['num_channel']
        vi_res = self.video_config['resolution']
        vi_num_channel = self.video_config['num_channel']

        a_reshape = tf.reshape(a, (-1, au_res1, au_res2, au_num_channel))
        v_reshape = tf.reshape(v, (-1, vi_res, vi_res, vi_num_channel))

        with tf.GradientTape(persistent=True) as tape:
            a_embedding = self.audio_encoder(
                a_reshape, training=audio_encoder_training,
                return_sequence=True)

            v_embedding = self.video_encoder(
                v_reshape, training=video_encoder_training,
                return_sequence=True)

            fusion_embedding, attention_weights = self.fusion_model(
                a_embedding, v_embedding)

            pred = self.classifier(
                fusion_embedding, training=True)

            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=target))

        if audio_encoder_training:
            g_ae = tape.gradient(loss, self.audio_encoder.trainable_variables)
            self.optimizer.apply_gradients(zip(g_ae, self.audio_encoder.trainable_variables))

        if video_encoder_training:
            g_ve = tape.gradient(loss, self.video_encoder.trainable_variables)
            self.optimizer.apply_gradients(zip(g_ve, self.video_encoder.trainable_variables))

        g_f = tape.gradient(loss, self.fusion_model.trainable_variables)
        self.optimizer.apply_gradients(zip(g_f, self.fusion_model.trainable_variables))

        g_c = tape.gradient(loss, self.classifier.trainable_variables)
        self.optimizer.apply_gradients(zip(g_c, self.classifier.trainable_variables))

        del tape

        self.train_loss(target, pred)
        self.train_acc(target, tf.nn.softmax(pred))

    def evaluate_full_model(self, val_dataset, epoch):

        for (batch, ((v, a), target)) in enumerate(val_dataset):
            au_res1 = self.audio_config['res1']
            au_res2 = self.audio_config['res2']
            au_num_channel = self.audio_config['num_channel']
            vi_res = self.video_config['resolution']
            vi_num_channel = self.video_config['num_channel']

            a_reshape = tf.reshape(a, (-1, au_res1, au_res2, au_num_channel))
            v_reshape = tf.reshape(v, (-1, vi_res, vi_res, vi_num_channel))

            a_embedding = self.audio_encoder(
                a_reshape, training=False,
                return_sequence=True)

            v_embedding = self.video_encoder(
                v_reshape, training=False,
                return_sequence=True)

            fusion_embedding, attention_weights = self.fusion_model(
                a_embedding, v_embedding)

            pred = self.classifier(
                fusion_embedding, training=False)

            self.val_loss(target, pred)
            self.val_acc(target, tf.nn.softmax(pred))

        print('Validation: Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
            epoch + 1, self.val_loss.result(), self.val_acc.result()
        ))

    def train_audio_encoder(self, EPOCHS, batch_size, test=False, audio_save_path=None):

        train_dataset, val_dataset = make_audio_data(
            self.audio_config, batch_size=batch_size, test=test)

        train_log = {
            'train': {'loss': [], 'accuracy': []},
            'val': {'loss': [], 'accuracy': []}
        }

        for epoch in range(EPOCHS):

            self.train_loss.reset_states()
            self.train_acc.reset_states()
            self.val_loss.reset_states()
            self.val_acc.reset_states()

            for (batch, (inp, target)) in enumerate(train_dataset):
                self._train_audio_encoder_step(inp, target)

                train_log['train']['loss'].append(self.train_loss.result())
                train_log['train']['accuracy'].append(self.train_acc.result())

                if batch % 5 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result(), self.train_acc.result()
                    ))
            self.evaluate_model(val_dataset, self.audio_encoder, epoch, self.audio_config)

            train_log['val']['loss'].append(self.val_loss.result())
            train_log['val']['accuracy'].append(self.val_acc.result())

        with open('audio_encoder_train_history.pickle', 'wb') as f:
            pickle.dump(train_log, f)

        self.audio_encoder.trained = True
        del train_dataset
        del val_dataset

        if audio_save_path is not None:
            self.audio_encoder.save(audio_save_path)

    def _train_audio_encoder_step(self, inp, target):

        res1 = self.audio_config['res1']
        res2 = self.audio_config['res2']
        num_channel = self.audio_config['num_channel']

        inp = tf.reshape(inp, (-1, res1, res2, num_channel))
        with tf.GradientTape() as tape:
            pred = self.audio_encoder(inp, training=True)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=target))
        grads = tape.gradient(loss, self.audio_encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.audio_encoder.trainable_variables))

        self.train_loss(target, pred)
        self.train_acc(target, tf.nn.softmax(pred))

    def train_video_encoder(self, EPOCHS, batch_size, test=False, video_save_path=None):

        train_dataset, val_dataset = make_video_data(
            self.video_config, batch_size=batch_size,
            test=test
        )

        train_log = {
            'train': {'loss': [], 'accuracy': []},
            'val': {'loss': [], 'accuracy': []}
        }

        for epoch in range(EPOCHS):

            self.train_loss.reset_states()
            self.train_acc.reset_states()
            self.val_loss.reset_states()
            self.val_acc.reset_states()

            for (batch, (inp, target)) in enumerate(train_dataset):
                self._train_video_encoder_step(inp, target)

                train_log['train']['loss'].append(self.train_loss.result())
                train_log['train']['accuracy'].append(self.train_acc.result())

                if batch % 5 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result(), self.train_acc.result()
                    ))
            self.evaluate_model(val_dataset, self.video_encoder, epoch, self.video_config)

            train_log['val']['loss'].append(self.val_loss.result())
            train_log['val']['accuracy'].append(self.val_acc.result())

        with open('video_encoder_train_history.pickle', 'wb') as f:
            pickle.dump(train_log, f)

        self.video_encoder.trained = True
        del train_dataset
        del val_dataset

        if video_save_path is not None:
            self.video_encoder.save(video_save_path)

    def _train_video_encoder_step(self, inp, target):

        resolution = self.video_config['resolution']
        num_channel = self.video_config['num_channel']

        inp = tf.reshape(inp, (-1, resolution, resolution, num_channel))
        with tf.GradientTape() as tape:
            pred = self.video_encoder(inp, training=True)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=target))
        grads = tape.gradient(loss, self.video_encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.video_encoder.trainable_variables))

        self.train_loss(target, pred)
        self.train_acc(target, tf.nn.softmax(pred))

    def evaluate_model(self, val_dataset, model, epoch, config):

        if 'res1' in config:
            res1 = config['res1']
            res2 = config['res2']
        else:
            res1 = res2 = config['resolution']
        num_channel = config['num_channel']

        for (batch, (inp, target)) in enumerate(val_dataset):
            X_test = tf.reshape(inp, (-1, res1, res2, num_channel))
            pred = model(X_test, training=False)
            self.val_loss(target, pred)
            self.val_acc(target, tf.nn.softmax(pred))

        print('Validation: Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
            epoch + 1, self.val_loss.result(), self.val_acc.result()
        ))

    def predict(self, inp, model_type=None):

        emotion = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        if model_type == 'video':
            n_frames = video_config['n_frames']
            resolution = video_config['resolution']
            speech_filename = video_config['speech_filename']

            cap = cv2.VideoCapture(inp)
            i = 0
            imgs = []
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                if i % int(cap.get(7) / n_frames) == 0:
                    img = cv2.resize(frame[x_start:x_end, y_start:y_end, :], frame_size)
                    if norm:
                        img = img / 255.0
                    imgs.append(img)
                i += 1
            cap.release()
            imgs_video = imgs_video(imgs[:n_frames])
            imgs = imgs_video.reshape([-1] + list(imgs_video.shape)[-3:])
            imgs = preprocess_input(imgs)
            imgs = imgs.reshape((-1, n_frames, resolution, resolution, 3))

            index = tf.math.argmax(tf.nn.softmax(self.video_encoder(imgs, training=False)))

        else:
            res1 = self.audio_config['res1']
            res2 = self.audio_config['res2']
            num_channel = self.audio_config['num_channel']

            inp = tf.convert_to_tensor(np.squeeze(wavfile_to_examples(inp).detach().numpy()))
            inp = tf.reshape(inp, (-1, res1, res2, num_channel))
            index = tf.math.argmax(tf.nn.softmax(self.audio_encoder(inp, training=False)), axis=1)

        print(emotion[int(index)])


class VideoEncoder(tf.keras.Model):
    def __init__(self, num_label, seq_length, d_embedding):
        super(VideoEncoder, self).__init__()

        self.resnet = ResNet50(include_top=False, weights='imagenet', pooling='max')
        self.resnet.trainable = None
        self.trained = False

        self.seq_length = seq_length
        self.d_embedding = d_embedding

        self.globalmaxpool = tf.keras.layers.GlobalMaxPooling1D()
        self.d_embedding = d_embedding
        self.dense_0 = tf.keras.layers.Dense(d_embedding, activation='relu')
        self.dense_1 = tf.keras.layers.Dense(num_label)

    def call(self, x, training, return_sequence=None, return_embedding=None):

        self.resnet.trainable = training

        x = self.resnet(x, training=self.resnet.trainable)
        x = self.dense_0(x)
        seq_embedding = tf.reshape(x, (-1, self.seq_length, self.d_embedding))
        if return_sequence:
            return seq_embedding
        embedding = self.globalmaxpool(seq_embedding)
        if return_embedding:
            return embedding
        y = self.dense_1(embedding)

        return y


class AudioEncoder(tf.keras.Model):
    def __init__(self, vgg_cfg, num_label, d_embedding, seq_length=5):
        super(AudioEncoder, self).__init__()
        self.seq_length = seq_length
        self.d_embedding = d_embedding

        self.vggish = self._vggish(vgg_cfg)
        self.vggish.trainable = None
        self.trained = False

        self.flatten = tf.keras.layers.Flatten()
        self.dropout_0 = tf.keras.layers.Dropout(0.3)
        self.dense_0 = tf.keras.layers.Dense(4096, activation='relu')
        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dense_1 = tf.keras.layers.Dense(4096, activation='relu')
        self.dropout_2 = tf.keras.layers.Dropout(0.3)
        self.dense_2 = tf.keras.layers.Dense(d_embedding, activation='relu')
        self.globalmaxpool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense_3 = tf.keras.layers.Dense(num_label)

    def call(self, x, training, return_sequence=None, return_embedding=None):

        self.vggish.trainable = training

        x = self.vggish(x)
        x = self.flatten(x)
        x = self.dropout_0(x, training=training)
        x = self.dense_0(x)
        x = self.dropout_1(x, training=training)
        x = self.dense_1(x)
        x = self.dropout_2(x, training=training)
        x = self.dense_2(x)
        seq_embedding = tf.reshape(x, (-1, self.seq_length, self.d_embedding))
        if return_sequence:
            return seq_embedding
        embedding = self.globalmaxpool(seq_embedding)
        if return_embedding:
            return embedding
        y = self.dense_3(embedding)

        return y

    def _vggish(self, cfg):
        feature_layers = []
        for v in cfg:
            if v == "M":
                feature_layers.append(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
            else:
                conv2d = tf.keras.layers.Conv2D(v, kernel_size=3, padding='same', activation="relu")
                feature_layers.append(conv2d)

        return tf.keras.models.Sequential(feature_layers, name="feature")


class FusionNetwork(tf.keras.Model):
    def __init__(self, seq_length, d_embedding):
        super(FusionNetwork, self).__init__()

        self.seq_length = seq_length
        self.d_embedding = d_embedding

        self.wq = tf.keras.layers.Dense(d_embedding)
        self.wk = tf.keras.layers.Dense(d_embedding)
        self.wv = tf.keras.layers.Dense(2 * d_embedding)

    def call(self, au, vi):
        v = tf.concat([vi, au], 2)

        vi = self.wq(vi)
        au = self.wk(au)
        v = self.wv(v)

        au_vi = tf.matmul(vi, au, transpose_b=True)
        diag = tf.linalg.diag_part(au_vi)
        score = diag / tf.math.sqrt(tf.cast(self.d_embedding, tf.float32))
        weights = tf.reshape(tf.nn.softmax(score, axis=-1), (-1, self.seq_length, 1))
        output = tf.matmul(weights, v, transpose_a=True)

        return tf.squeeze(output), tf.squeeze(weights)


class DenseClassifier(tf.keras.Model):
    def __init__(self, num_label):
        super(DenseClassifier, self).__init__()

        self.dense_0 = tf.keras.layers.Dense(2048, activation='relu')
        self.dropout_0 = tf.keras.layers.Dropout(0.2)
        self.dense_1 = tf.keras.layers.Dense(2048, activation='relu')
        self.dropout_1 = tf.keras.layers.Dropout(0.2)
        self.dense_2 = tf.keras.layers.Dense(num_label)

    def call(self, x, training):
        x = self.dense_0(x)
        x = self.dropout_0(x, training=training)
        x = self.dense_1(x)
        x = self.dropout_1(x, training=training)
        y = self.dense_2(x)

        return y


if __name__ == '__main__':
    video_config = {
        'n_frames': 5,
        'resolution': 64,
        'speech_filename': 'Video_Speech',
        'num_channel': 3
    }
    audio_config = {
        'speech_filename': 'outputdata',
        'res1': 96,
        'res2': 64,
        'num_channel': 1
    }

    cfgs = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M']

    full_model = MultiModalModel(
        video_config, audio_config,
        num_label=8, seq_length=5,
        d_embedding=512, vgg_cfg=cfgs
    )

    full_model.train_full_model(
        EPOCHS=40, batch_size=32,
        video_encoder_training=False,
        audio_encoder_training=False
    )

    full_model.save('full-MMM')