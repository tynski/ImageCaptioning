import json
import os
import pickle
import random
import time
import numpy as np
import tensorflow as tf

from image_captioning.data.data_processing import prepare_img_tensor, load_features, load_image
from image_captioning.models.decoder import RNN_Decoder
from image_captioning.models.encoder import CNN_Encoder


checkpoint_path = 'image_captioning/training_checkpoints/'
checkpoint_prefix = os.path.join(checkpoint_path, 'ckpt')
results_training_path = 'results/training/'
results_validation_path = 'results/validation/'
repository_path = 'image_captioning/data/repository.pkl'

embedding_dim = 256
units = 512


class Model:
    def __init__(self, top_k, batch_size):
        self.top_k = top_k
        self.batch_size = batch_size
        self.loss_object = None

        self.encoder = CNN_Encoder(embedding_dim)
        self.decoder = RNN_Decoder(embedding_dim, units, self.top_k + 1)
        self.optimizer = tf.keras.optimizers.Adam()

        self.models_trained = False

        self.checkpoint = tf.train.Checkpoint(
            encoder=self.encoder, decoder=self.decoder, optimizer=self.optimizer)

        try:
            with open(repository_path, 'rb') as handle:
                self.repository = pickle.load(handle)
        except FileNotFoundError:
            print('Can not found repository file! Have you prepared it?')

        if tf.train.latest_checkpoint(checkpoint_path):
            print('Restoring last checkpoint.')

            self.checkpoint.restore(
                tf.train.latest_checkpoint(checkpoint_path))
            self.models_trained = True
        else:
            self.repository['start_epoch'] = 0

    def train(self, n_samples, epochs):
        start_epoch = self.repository['start_epoch']

        if start_epoch >= epochs:
            return 'Trained enough! Start epoch: {}'.format(start_epoch)

        print('Training started!')

        features_path_list = self.repository['train_feature_path_list']
        caption_list = self.repository['train_caption_list']
        tokenizer = self.repository['tokenizer']
        batches_num = n_samples // self.batch_size
        loss_plot = []

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        for epoch in range(start_epoch, epochs):
            start = time.time()
            total_loss = 0
            batch_pointer = [i for i in range(batches_num)]
            random.shuffle(batch_pointer)

            for i in range(batches_num):
                batch_start = batch_pointer[i] * self.batch_size
                batch_end = batch_start + self.batch_size
                img_tensor = prepare_img_tensor(
                    features_path_list[batch_start:batch_end], self.batch_size)
                caption_tensor = tf.convert_to_tensor(
                    caption_list[batch_start:batch_end], dtype=tf.float32)
                batch_loss, t_loss = self.train_step(
                    img_tensor, caption_tensor, tokenizer)
                total_loss += t_loss

                if i % 100 == 0:
                    train_progress = 'Epoch {} Batch {} Loss {:.4f}\n'.format(
                        epoch + 1, i, batch_loss.numpy() / int(caption_tensor.shape[1]))

                    print(train_progress)

                    with open(results_training_path + 'loss_history.txt', 'a+') as file:
                        file.write(train_progress)

            loss_plot.append(total_loss / batches_num)

            self.checkpoint.save(file_prefix=checkpoint_prefix)

            epoch_progress = 'Epoch {} Loss {:.6f}\nTime taken for 1 epoch {} sec\n'.format(
                epoch + 1, total_loss / batches_num, time.time() - start)

            print(epoch_progress)

            with open(results_training_path + 'loss_history.txt', 'a+') as file:
                file.write(epoch_progress)

        with open(results_training_path + 'loss_plot.pkl', 'a+b') as file:
            pickle.dump(loss_plot, file)

        print('Training finished!')
        
        self.models_trained = True
        self.repository['start_epoch'] += epoch-start_epoch+1
        self.repository['tokenizer'] = tokenizer

        with open(repository_path, 'w+b') as file:
            pickle.dump(self.repository, file)

    def validate(self, n_samples):
        if not self.models_trained:
            raise NotImplementedError(
                'Models are not trained. Traine Them first!')

        print('Validation started!')

        img_id_list = self.repository['val_img_id_list']
        features_path_list = self.repository['val_feature_path_list']
        tokenizer = self.repository['tokenizer']
        max_length = self.repository['max_length']
        
        results = []
        i = 0

        for i in range(n_samples):
            image_id = img_id_list[i]
            features_path = features_path_list[i]
            features = load_features(features_path)
            features_tensor = tf.reshape(features, (1, 64, 2048))
            features = self.encoder(features_tensor)
            prediction = self.predict_caption(
                features, self.decoder, tokenizer, max_length)
            results.append({'image_id': image_id, 'caption': ' '.join(prediction)})

        with open('{}captions_val2014_results.json'.format(results_validation_path), 'w+') as write_file:
            json.dump(results, write_file)

        print('Validataion finished!')

    def test(self, n_samples):
            if not self.models_trained:
                raise NotImplementedError(
                    'Models are not trained. Traine Them first!')

            print('Test started!')

            img_id_list = self.repository['test_img_id_list']
            features_path_list = self.repository['test_feature_path_list']
            tokenizer = self.repository['tokenizer']
            max_length = self.repository['max_length']
            results = []
            i = 0

            for i in range(n_samples):
                image_id = img_id_list[i]
                features_path = features_path_list[i]
                features = load_features(features_path)
                features_tensor = tf.reshape(features, (1, 64, 2048))
                features = self.encoder(features_tensor)
                prediction = self.predict_caption(
                    features, self.decoder, tokenizer, max_length)
                results.append({'image_id': image_id, 'caption': ' '.join(prediction)})

            with open('{}captions_test2014_results.json'.format(results_validation_path), 'w+') as write_file:
                json.dump(results, write_file)

            print('Test finished!')


    def evaluate(self, image, img_id, image_features_extract_model):
        if not self.models_trained:
            raise NotImplementedError(
                'Models are not trained. Traine Them first!')

        print('Evaluating image')

        tokenizer = self.repository['tokenizer']
        max_length = self.repository['max_length']
        attention_plot = np.zeros((max_length, 64))
        hidden = self.decoder.reset_state(batch_size=1)
        temp_input = tf.expand_dims(load_image(image), 0)
        img_tensor_val = image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (64, 2048))
        features = self.encoder(img_tensor_val)
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(max_length):
            predictions, hidden, attention_weights = self.decoder(
                dec_input, features, hidden)
            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()

            try:
                word = tokenizer.index_word[predicted_id]
            except KeyError:
                word = '<unk>'

            if word == '<end>':
                break

            result.append(word)
            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        timestamp = time.strftime('%H%M%S')

        with open('{}attention_plot-{}.pkl'.format(results_validation_path, img_id), 'w+b') as file:
            pickle.dump(attention_plot, file)

        with open('{}prediction-{}.json'.format(results_validation_path, img_id), 'w+') as file:
            json.dump(result, file)

        print('Evaluation finished')

    @staticmethod
    def predict_caption(features, decoder, tokenizer, max_length):
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []
        hidden = decoder.reset_state(batch_size=1)

        i = 0
        while i < max_length:
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()

            try:
                word = tokenizer.index_word[predicted_id]
            except KeyError:
                word = '<unk>'

            if word == '<end>':
                break

            result.append(word)
            dec_input = tf.expand_dims([predicted_id], 0)
            i += 1

        return result

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, img_tensor, target, tokenizer):
        loss = 0
        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = self.decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims(
            [tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = self.decoder(
                    dec_input, features, hidden)

                loss += self.loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))
        trainable_variables = self.encoder.trainable_variables + \
            self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss
