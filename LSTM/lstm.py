from __future__ import print_function
from etaprogress.progress import ProgressBar
import tensorflow as tf
import sys, os
import time
import numpy as np
import re


batch_size=128

class Dataset():
    def __init__(self):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._read_data()
        self._split_dataset()

    def _read_data(self):
        feat_root = 'data/feats/raw_mfcc'
        raw_data = []
        self._max_length = 0
        for i in range(1, 9):
            feat_path = feat_root + str(i) + '.txt'
            file = open(feat_path)
            content = file.read()
            all_list = re.findall(r'\[.+?\]', content, flags=re.DOTALL)
            #names = re.findall(r'\].*?(\w+).*?\[', content, flags=re.DOTALL)

            for record in all_list:
                numbers = re.findall(r'[e\d\.-]+', record)
                if len(numbers) > self._max_length:
                    self._max_length = len(numbers)
                raw_data.append(numbers)
        self._max_length //= 13
        self._data = np.zeros([len(raw_data), self._max_length, 13])
        self._seq_len = np.zeros(len(raw_data), dtype=np.int32)
        for idx, sequence in enumerate(raw_data):
            sequence_nparray = np.array(list(map(float, sequence)))
            sequence_nparray = sequence_nparray.reshape([-1,13])
            self._data[idx, :sequence_nparray.shape[0]] = sequence_nparray
            self._seq_len[idx] = sequence_nparray.shape[0]

        ali_root = 'data/ali/ali'
        self._label = np.zeros([self._data.shape[0], self._data.shape[1], 19])
        sequence_id = 0
        for i in range(1, 17):
            ali_path = ali_root + str(i) + '.txt'
            file = open(ali_path)
            for line in file.readlines():
                numbers = line.split()
                for idx, number in enumerate(numbers[1:]): #ignore sequence name
                    self._label[sequence_id, idx, int(number)-1] = 1
                sequence_id += 1

    def _split_dataset(self):
        division_ratio = 0.9
        division_pos = int(self._data.shape[0] * division_ratio)
        self.train_data = self._data[:division_pos]
        self.test_data = self._data[division_pos:]
        self.train_label, self.test_label = self._label[:division_pos], self._label[division_pos:]
        self.train_seq_len, self.test_seq_len = self._seq_len[:division_pos], self._seq_len[division_pos:]
        self._num_train_examples = self.train_data.shape[0]
        self._num_test_examples = self.test_data.shape[0]

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_train_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_train_examples,dtype="int32")
            np.random.shuffle(perm)
            self.train_data = self.train_data[perm]
            self.train_label = self.train_label[perm]
            self._index_in_epoch = batch_size
            start = 0
            assert batch_size <= self._num_train_examples
        end = self._index_in_epoch
        return self.train_data[start:end], self.train_label[start:end], self.train_seq_len[start:end]

    def get_test_data(self):
        return self.test_data, self.test_label, self.test_seq_len
    def get_test_size(self):
        return self._num_test_examples

    def get_max_len(self):
        return self._max_length


class TensorflowLSTM():
    def __init__(self, h_size=128, n_inputs=16, n_steps=260, n_classes=19, l_r=0.001, test_size=128):
        # parameters init
        l_r = l_r
        self.n_inputs = n_inputs
        self.n_steps = n_steps
        n_classes = n_classes
        self.model_dir = 'model/tf/lstm'

        ## build graph
        tf.reset_default_graph()
        self.X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='X')
        self.Y = tf.placeholder(tf.float32, [None, n_steps, n_classes], name='Y')
        self.Seq_len = tf.placeholder(tf.int32, [None], name='Seq_len')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')

        w1 = tf.Variable(tf.random_normal([h_size, n_classes]))
        b1 = tf.Variable(tf.random_normal([1, n_classes]))

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(h_size,activation=tf.nn.sigmoid)
        init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, self.X, initial_state=init_state) #, sequence_length=self.Seq_len
        #outputs (N, n_steps, h_size)
        pred = tf.matmul(tf.reshape(outputs, [-1, h_size]), w1) + b1
        pred_3D = tf.reshape(pred, [self.batch_size, n_steps, n_classes])
        self.cost = tf.Variable(0.)
        for indx in range(batch_size):
            logits = pred_3D[indx, :self.Seq_len[indx], :]
            labels = self.Y[indx, :self.Seq_len[indx], :]
            #logits_flat = tf.reshape(logits, [-1, n_classes])
            #labels_flat = tf.reshape(labels, [-1, n_classes])
            self.cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        self.cost /= tf.cast(self.batch_size, tf.float32)
        #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.Y))
        tf.summary.scalar('loss', self.cost)
        self.train_op = tf.train.AdamOptimizer(l_r).minimize(self.cost)

        self.accuracy = tf.Variable(0.)
        for indx in range(test_size):
            _pred = pred_3D[indx, :self.Seq_len[indx], :]
            _Y = self.Y[indx, :self.Seq_len[indx], :]
            _correct_pred = tf.equal(tf.argmax(_pred, 1),tf.argmax(_Y, 1))
            self.accuracy += tf.reduce_mean(tf.cast(_correct_pred, tf.float32))
        self.accuracy /= float(test_size)
        #self.correct_pred = tf.equal(tf.argmax(pred_3D, 2),tf.argmax(self.Y, 2))
        #self.correct_pred_float = tf.cast(self.correct_pred,tf.float32)
        #self.accuracy = tf.reduce_mean(self.correct_pred_float)
        #tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def fit(self, dataset, n_epoch=100, epoch_size = 5000):

        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        test_x, test_y, test_seq_len = dataset.get_test_data()
        acc = 0
        with tf.Session(config=config) as sess:
            #shutil.rmtree('/tmp/TF/MNIST')
            self.sw_train = tf.summary.FileWriter('/tmp/TF/MNIST/train', sess.graph)
            self.sw_test = tf.summary.FileWriter('/tmp/TF/MNIST/test')
            sess.run(init_op)
            for i in range(n_epoch):
                print('Epoch %d/%d' % (i+1, n_epoch))
                bar = ProgressBar(int(epoch_size/batch_size), max_width=80)
                for j in range(int(epoch_size/batch_size)):
                    batch_x, batch_y, batch_seq_len = dataset.next_batch(batch_size)
                    assert batch_x.shape[0] == batch_size and batch_y.shape[0] == batch_size and batch_seq_len.shape[0] == batch_size
                    summary, _, cost = sess.run([self.merged, self.train_op, self.cost],
                            feed_dict={self.X: batch_x, self.Y: batch_y, self.Seq_len:batch_seq_len, self.batch_size: batch_size})
                    self.sw_train.add_summary(summary, i*int(epoch_size/batch_size) + j)
                    bar.numerator = j+1
                    print("%s | loss: %f | test_acc: %.2f" % (bar, cost, acc*100), end='\r')
                    sys.stdout.flush()
                    if j % 100 == 0:
                        summary, cost, acc = sess.run([self.merged, self.cost, self.accuracy],
                                feed_dict={self.X: test_x, self.Y: test_y,  self.Seq_len:test_seq_len, self.batch_size: len(test_x)})
                        self.sw_test.add_summary(summary, i*int(epoch_size/batch_size)+j)
                print()
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            saver = tf.train.Saver()
            save_path = saver.save(sess,'%s/model.ckpt' % self.model_dir)
            print("Model saved in file: %s" % save_path)

    '''def predict(self, X_test):
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        assert np.shape(X_test)[1:] == (28, 28)
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
            return sess.run(self.pred, feed_dict={self.X: X_test, self.batch_size: len(X_test)})'''

def main():
    dataset = Dataset()

    tf_lstm = TensorflowLSTM(h_size=128, n_inputs=13, n_steps=dataset.get_max_len(), n_classes=19, l_r=0.001, test_size=dataset.get_test_size())
    t1 = time.time()
    tf_lstm.fit(dataset)
    t2 = time.time()
    '''print('training time: %s' % (t2-t1))
    pred = tf_lstm.predict(mnist.test.images.reshape(-1, 28, 28))
    t3 = time.time()
    print('predict time: %s' % (t3-t2))
    test_lab = mnist.test.labels
    print("accuracy: ", np.mean(np.equal(np.argmax(pred,1), np.argmax(test_lab,1)))*100)'''


if __name__ == '__main__':
    main()