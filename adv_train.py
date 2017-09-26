from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import re
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical

from model import Model
from data_helper import load_data, DataReader

flags = tf.flags

# data
flags.DEFINE_string('data_dir', './data', 'data directory. Should contain train.txt/valid.txt/test.txt with input data')

# model params
flags.DEFINE_integer('rnn_size', 250, 'size of LSTM internal state')
flags.DEFINE_integer('adv_l', 0.7, 'size of LSTM internal state')
flags.DEFINE_integer('highway_layers', 0, 'number of highway layers')
flags.DEFINE_integer('char_embed_size', 25, 'dimensionality of character embeddings')
flags.DEFINE_string('kernels', '[1,2,3,4,5,6]', 'CNN kernel widths')
flags.DEFINE_string('kernel_features', '[50,50,100,100,200,200]', 'number of features in the CNN kernel')
flags.DEFINE_integer('num_classes',53,'25 classed in sum ,but eos if another class')

# optimization
flags.DEFINE_float('learning_rate', 0.0001, 'starting learning rate')
flags.DEFINE_float('learning_rate_decay', 1, 'learning rate decay')
flags.DEFINE_float('decay_when', 1.0, 'decay if validation perplexity does not improve by more than this much')
flags.DEFINE_float('param_init', 0.05, 'initialize parameters at')
flags.DEFINE_integer('num_unroll_steps', 39, 'number of timesteps to unroll for')
flags.DEFINE_integer('batch_size', 20, 'number of sequences to train on in parallel')
flags.DEFINE_integer('max_epochs', 100, 'number of full passes through the training data')
flags.DEFINE_integer('max_word_length', 35, 'maximum word length')

# bookkeeping
flags.DEFINE_integer('seed', 3435, 'random number generator seed')
flags.DEFINE_integer('print_every', 20, 'how often to print current loss')
flags.DEFINE_string('EOS', '+',
                    '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')

FLAGS = flags.FLAGS


def main():
    dataname = ['train', 'test', 'dev', 'domain_ptb', 'domain_twe']
    word_vocab, char_vocab, label_vocab, word_tensors, char_tensors, label_tensors, mask_tensors = \
        load_data(FLAGS.data_dir, dataname, FLAGS.num_unroll_steps, FLAGS.max_word_length)
    # word_tensor, char_tensor,label_tensor, batch_size, num_class, num_unroll_steps
    train_reader = DataReader(word_tensors['train'], char_tensors['train'], label_tensors['train'],
                              mask_tensors['train'],
                              FLAGS.batch_size, FLAGS.num_classes)
    test_reader = DataReader(word_tensors['test'], char_tensors['test'], label_tensors['test'], mask_tensors['test'],
                             FLAGS.batch_size, FLAGS.num_classes)
    dev_reader = DataReader(word_tensors['dev'], char_tensors['dev'], label_tensors['dev'], mask_tensors['dev'],
                            FLAGS.batch_size, FLAGS.num_classes)
    domainptb_reader = DataReader(word_tensors['domain_ptb'], char_tensors['domain_ptb'], label_tensors['domain_ptb'],
                                  mask_tensors['domain_ptb'],
                                  FLAGS.batch_size, FLAGS.num_classes)
    domaintwe_reader = DataReader(word_tensors['domain_twe'], char_tensors['domain_twe'], label_tensors['domain_twe'],
                                  mask_tensors['domain_twe'],
                                  FLAGS.batch_size, FLAGS.num_classes)

    print('initialized all dataset readers')

    args = FLAGS
    args.char_vocab_size = char_vocab.size
    args.word_vocab_size = word_vocab.size
    args.word_vocab = word_vocab

    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        ''' build training graph '''
        # para_init=0.05
        initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
        with tf.variable_scope("Model", initializer=initializer):
            train_model = Model(args)
    with tf.Session(graph=g) as session:
        tf.initialize_all_variables().run()
        print('Created and initialized fresh model.')


        ''' training starts here '''
        twitter_label = []
        ptb_label = []

        # add domian label
        for i in range(args.batch_size * args.num_unroll_steps):
            twitter_label.append([1, 0])
            ptb_label.append([0, 1])

        model_path = "./adv_model.ckpt"
        saver = tf.train.Saver()
        valid_best_accuracy = 0

        # training start
        for epoch in range(10):
            domainptb_iter = domainptb_reader.iter()
            domaintwe_iter = domaintwe_reader.iter()

            for i in range(3200):

                x,y,z,m = next(domainptb_iter)
                sentence_length = []
                for batch in m:
                    sentence_length.append(sum(batch))

                class_loss, domain_loss, class_accuracy, domain_accuracy, _ = session.run(
                    [
                        # sequence order change
                        train_model.class_loss,
                        train_model.domain_loss,
                        train_model.class_accuracy,
                        train_model.domain_accuracy,
                        train_model.total_train_op
                    ], {
                        train_model.input_: x,
                        train_model.class_targets: y,
                        train_model.input_word: z,
                        train_model.input_mask: m,
                        train_model.sentence_length: sentence_length,
                        train_model.domain_targets: ptb_label,
                        train_model.dropout: 0.5,
                        train_model.learning_rate: FLAGS.learning_rate
                    })

                if i % (FLAGS.print_every*10) == 0:
                    print('the %d time of PTB class  loss is:%f'%(i,class_loss))
                    print('the %d time of PTB class accuracy is:%f' % (i, class_accuracy))
                    print('the %d time of PTB domain loss is:%f' % (i, domain_loss))
                    print('the %d time of PTB domain accuracy is:%f' % (i, domain_accuracy))

                # print('**************** NOW INSERT UNlabel TWE ***********************')
                x, y, z, m = next(domaintwe_iter)
                sentence_length = []
                for batch in m:
                    sentence_length.append(sum(batch))
                class_loss, domain_loss, autoencoder_cost, class_accuracy, domain_accuracy, _ = session.run(
                    [
                        # sequence order change
                        train_model.class_loss,
                        train_model.domain_loss,
                        train_model.autoencoder_loss,
                        train_model.class_accuracy,
                        train_model.domain_accuracy,
                        train_model.domain_train_op
                    ], {
                        train_model.input_: x,
                        train_model.class_targets: y,
                        train_model.input_word: z,
                        train_model.input_mask: m,
                        train_model.sentence_length: sentence_length,
                        train_model.domain_targets: twitter_label,
                        train_model.dropout: 0.5,
                        train_model.learning_rate: FLAGS.learning_rate
                    })

                if i % (FLAGS.print_every * 10) == 0:
                    print('the %d time of TWE class loss is:%f' % (i, class_loss))
                    print('the %d time of TWE class accuracy is:%f' % (i, class_accuracy))
                    print('the %d time of TWE domain loss is:%f' % (i, domain_loss))
                    print('the %d time of TWE autoencoder_cost is:%f' % (i, autoencoder_cost))
                    print('the %d time of TWE domain accuracy is:%f' % (i, domain_accuracy))



            print('^^^^^^^^^^  VALIDATION BEGIN  ^^^^^^^^^^^^')
            avg_valid_loss = []
            avg_valid_correct = []
            avg_valid_sum = []
            # rnn_state = session.run(test_model.initial_rnn_state)

            for x, y, z, m in train_reader.iter():
                sentence_length = []
                for batch in m:
                    sentence_length.append(sum(batch))
                loss, correct_num, sum_num = session.run([
                    train_model.class_loss,
                    train_model.correct_num,
                    train_model.sum_num
                ], {
                    train_model.input_: x,
                    train_model.class_targets: y,
                    train_model.input_word: z,
                    train_model.input_mask: m,
                    train_model.sentence_length: sentence_length,
                    train_model.dropout: 1
                })

                avg_valid_loss.append(loss)
                avg_valid_correct.append(correct_num)
                avg_valid_sum.append(sum_num)


            print(
                'the %d testing epoch of average loss is:%f' % (epoch, sum(avg_valid_loss) / len(avg_valid_loss)))
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@the accuracy of %d VALIDATION epoch is %f:' % (
            epoch, sum(avg_valid_correct) / sum(avg_valid_sum)))
            target_valid_accuracy =  sum(avg_valid_correct) / sum(avg_valid_sum)
            if target_valid_accuracy>valid_best_accuracy:
                save_path = saver.save(session,model_path)
                valid_best_accuracy = target_valid_accuracy
                print("^^^^^^^^^^  SAVE MODEL !!!  ^^^^^^^^^^^^")
            else:
                print('^^^^^^^^^^  NOT SAVED !!!  ^^^^^^^^^^^^^')



        print('##############  NOW BEGIN FINE-TUNE !!!!!!  ############### ')
        # epoch done: time to evaluate
        saver = tf.train.Saver()
        load_path = saver.restore(session,model_path)

        for epoch in range(FLAGS.max_epochs):
            print('**************** NOW INSERT LABELED TWITTER ***********************')
            avg_train_loss = []
            avg_train_correct = []
            avg_train_sum = []
            count = 0

            for x, y, z, m in train_reader.iter():
                sentence_length = []
                for batch in m:
                    sentence_length.append(sum(batch))
                count += 1
                loss, correct_num, sum_num, _ = session.run([
                    # sequence order change
                    train_model.class_loss,
                    train_model.correct_num,
                    train_model.sum_num,
                    train_model.class_train_op
                ], {
                    train_model.input_: x,
                    train_model.class_targets: y,
                    train_model.input_word: z,
                    train_model.input_mask: m,
                    train_model.sentence_length: sentence_length,
                    train_model.dropout : 0.8,
                    train_model.learning_rate: FLAGS.learning_rate * (FLAGS.learning_rate_decay ** epoch)
                })
                avg_train_loss.append(loss)
                avg_train_correct.append(correct_num)
                avg_train_sum.append(sum_num)


            print('the %d training epoch of average loss is:%f'%(epoch,sum(avg_train_loss)/len(avg_train_loss)))
            print('the class accuracy of %d !!!!!!!!!!!!!training epoch is %f:'%(epoch,sum(avg_train_correct)/sum(avg_train_sum)))


            print('**************** NOW VALID ***********************')
            avg_dev_loss = []
            file = open('dev_text.txt','w')
            error_num=0
            for x, y, z, m in dev_reader.iter():
                sentence_length = []
                for batch in m:
                    sentence_length.append(sum(batch))
                loss, y_pred, train_index, test_index = session.run([
                    train_model.class_loss,
                    train_model.y_pred1,
                    train_model.train_index,
                    train_model.test_index
                ], {
                    train_model.input_: x,
                    train_model.class_targets: y,
                    train_model.input_word: z,
                    train_model.input_mask: m,
                    train_model.sentence_length: sentence_length,
                    train_model.dropout: 1
                })

                word_batch = z.flatten()

                # substitute the #@RTURL
                patternHT = '#[\w]+'
                patternUSR = '@[\w]+'
                patternURL = 'http|www\.|^com[^\w]'

                inputword = np.ndarray.flatten(z)
                # inputword[batch_size, time]
                y_predtrue = []
                # y_pred[batch_size * time, num_classes]
                for label, word in zip(y_pred, inputword):
                    if re.match(patternHT, word_vocab._index2token[word]):
                        y_predtrue.append(
                            to_categorical([label_vocab._token2index['HT']], num_classes=args.num_classes)[0])
                    elif re.match(patternURL, args.word_vocab._index2token[word]):
                        y_predtrue.append(
                            to_categorical([label_vocab._token2index['URL']], num_classes=args.num_classes)[0])
                    elif re.match(patternUSR, args.word_vocab._index2token[word]):
                        y_predtrue.append(
                            to_categorical([label_vocab._token2index['USR']], num_classes=args.num_classes)[0])
                    else:
                        label_index = np.argmax(label)
                        ht_index = label_vocab._token2index['HT']
                        url_index = label_vocab._token2index['URL']
                        usr_index = label_vocab._token2index['USR']
                        while (label_index == ht_index) or (label_index == url_index) or (label_index == usr_index):
                            label[label_index] = 0
                            label_index = np.argmax(label)
                        y_predtrue.append(label)
                y = np.reshape(y,(-1,args.num_classes))

                class_correct_prediction = np.equal(np.argmax(y_predtrue, 1), np.argmax(y, 1))

                for index,y_p,y,word in zip(class_correct_prediction,np.argmax(y_predtrue, 1),test_index,word_batch):
                    if index==False:
                        a=label_vocab._index2token[y_p]
                        b=label_vocab._index2token[y]
                        c=word_vocab._index2token[word]
                        if label_vocab._index2token[y]!='+':
                            error_num+=1

                            try:
                                file.write(c + '\t' + b + '\t' + a)
                                file.write('\n')
                            except Exception as e:
                                print(e)
                                pass



                avg_dev_loss.append(loss)
            file.close()

            print('the %d validation epoch of average loss is:%f' % (epoch, sum(avg_dev_loss) / len(avg_dev_loss)))
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@the true accuracy of %d validation epoch is %f:' % (
            epoch, (2242 - error_num) / 2242))

            if ((2242 - error_num) / 2242)>=valid_best_accuracy:
                save_path = saver.save(session,model_path)
                valid_best_accuracy = ((2242 - error_num) / 2242)
                print("^^^^^^^^^^  SAVE MODEL !!!  ^^^^^^^^^^^^")
            else:
                print('^^^^^^^^^^  NOT SAVED !!!  ^^^^^^^^^^^^^')


            #TESTING
            print('**************** NOW TESTING ***********************')

            avg_test_loss = []
            file = open('wrong_text.txt','w')
            error_num = 0
            for x, y, z, m in test_reader.iter():
                sentence_length = []
                for batch in m:
                    sentence_length.append(sum(batch))
                loss, y_pred,train_index,test_index= session.run([
                    train_model.class_loss,
                    train_model.y_pred1,
                    train_model.train_index,
                    train_model.test_index
                ], {
                    train_model.input_: x,
                    train_model.class_targets: y,
                    train_model.input_word: z,
                    train_model.input_mask: m,
                    train_model.sentence_length: sentence_length,
                    train_model.dropout: 1
                })
                word_batch = z.flatten()

                # substitute the #@RTURL
                patternHT = '#[\w]+'
                patternUSR = '@[\w]+'
                patternURL = 'http|www\.|^com[^\w]'

                inputword = np.ndarray.flatten(z)
                y_predtrue = []
                for label, word in zip(y_pred, inputword):
                    if re.match(patternHT, word_vocab._index2token[word]):
                        y_predtrue.append(
                            to_categorical([label_vocab._token2index['HT']], num_classes=args.num_classes)[0])
                    elif re.match(patternURL, args.word_vocab._index2token[word]):
                        y_predtrue.append(
                            to_categorical([label_vocab._token2index['URL']], num_classes=args.num_classes)[0])
                    elif re.match(patternUSR, args.word_vocab._index2token[word]):
                        y_predtrue.append(
                            to_categorical([label_vocab._token2index['USR']], num_classes=args.num_classes)[0])
                    else:
                        label_index = np.argmax(label)
                        ht_index = label_vocab._token2index['HT']
                        url_index = label_vocab._token2index['URL']
                        usr_index = label_vocab._token2index['USR']
                        while (label_index == ht_index) or (label_index == url_index) or (label_index == usr_index):
                            label[label_index] = 0
                            label_index = np.argmax(label)
                        y_predtrue.append(label)
                y = np.reshape(y, (-1, args.num_classes))

                class_correct_prediction = np.equal(np.argmax(y_predtrue, 1), np.argmax(y, 1))

                for index,y_p,y,word in zip(class_correct_prediction,np.argmax(y_predtrue, 1),test_index,word_batch):
                    if index==False:
                        a=label_vocab._index2token[y_p]
                        b=label_vocab._index2token[y]
                        c=word_vocab._index2token[word]
                        if label_vocab._index2token[y]!='+':
                            error_num+=1

                            try:
                                file.write(c + '\t' + b + '\t' + a)
                                file.write('\n')
                            except Exception as e:
                                print(e)
                                pass



                avg_test_loss.append(loss)

            file.close()


            print('the %d testing epoch of average loss is:%f' % (epoch, sum(avg_test_loss) / len(avg_test_loss)))
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@the true accuracy of %d testing epoch is %f:' % (epoch, (2291-error_num) / 2291))

def reproduce():
    dataname = ['train','test', 'dev','domain_ptb','domain_twe']
    word_vocab, char_vocab, label_vocab, word_tensors, char_tensors, label_tensors, mask_tensors = \
        load_data(FLAGS.data_dir, dataname, FLAGS.num_unroll_steps, FLAGS.max_word_length)

    test_reader = DataReader(word_tensors['test'], char_tensors['test'], label_tensors['test'], mask_tensors['test'],
                             FLAGS.batch_size, FLAGS.num_classes)
    dev_reader = DataReader(word_tensors['dev'], char_tensors['dev'], label_tensors['dev'], mask_tensors['dev'],
                            FLAGS.batch_size, FLAGS.num_classes)

    print('initialized all dataset readers')

    args = FLAGS
    args.char_vocab_size = char_vocab.size
    args.word_vocab_size = word_vocab.size
    args.word_vocab = word_vocab

    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        ''' build training graph '''
        # para_init=0.05
        initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
        with tf.variable_scope("Model", initializer=initializer):
            train_model = Model(args)


    with tf.Session(graph=g) as session:
        tf.initialize_all_variables().run()
        print('Created and initialized fresh model.')

        model_path = "./adv_model.ckpt"
        saver = tf.train.Saver()
        saver.restore(session, model_path)


        print('**************** NOW VALID ***********************')
        file = open('dev_results.txt','w')
        error_num=0
        for x, y, z, m in dev_reader.iter():
            sentence_length = []
            for batch in m:
                sentence_length.append(sum(batch))
            loss, y_pred, train_index, test_index = session.run([
                train_model.class_loss,
                train_model.y_pred1,
                train_model.train_index,
                train_model.test_index
            ], {
                train_model.input_: x,
                train_model.class_targets: y,
                train_model.input_word: z,
                train_model.input_mask: m,
                train_model.sentence_length: sentence_length,
                train_model.dropout: 1
            })

            word_batch = z.flatten()

            # substitute the #@RTURL
            patternHT = '#[\w]+'
            patternUSR = '@[\w]+'
            patternURL = 'http|www\.|^com[^\w]'

            inputword = np.ndarray.flatten(z)
            # inputword[batch_size, time]
            y_predtrue = []
            # y_pred[batch_size * time, num_classes]
            for label, word in zip(y_pred, inputword):
                if re.match(patternHT, word_vocab._index2token[word]):
                    y_predtrue.append(
                        to_categorical([label_vocab._token2index['HT']], num_classes=args.num_classes)[0])
                elif re.match(patternURL, args.word_vocab._index2token[word]):
                    y_predtrue.append(
                        to_categorical([label_vocab._token2index['URL']], num_classes=args.num_classes)[0])
                elif re.match(patternUSR, args.word_vocab._index2token[word]):
                    y_predtrue.append(
                        to_categorical([label_vocab._token2index['USR']], num_classes=args.num_classes)[0])
                else:
                    label_index = np.argmax(label)
                    ht_index = label_vocab._token2index['HT']
                    url_index = label_vocab._token2index['URL']
                    usr_index = label_vocab._token2index['USR']
                    while (label_index == ht_index) or (label_index == url_index) or (label_index == usr_index):
                        label[label_index] = 0
                        label_index = np.argmax(label)
                    y_predtrue.append(label)
            y = np.reshape(y,(-1,args.num_classes))

            class_correct_prediction = np.equal(np.argmax(y_predtrue, 1), np.argmax(y, 1))

            for index,y_p,y,word in zip(class_correct_prediction,np.argmax(y_predtrue, 1),test_index,word_batch):
                if label_vocab._index2token[y] != '+':
                    a = label_vocab._index2token[y_p]
                    b = label_vocab._index2token[y]
                    c = word_vocab._index2token[word]
                    file.write(c + '\t' + b + '\t' + a)
                    file.write('\n')

                    if index==False:
                            error_num+=1

        file.close()

        print('the accuracy of validation epoch is %f:'% ((2242 - error_num) / 2242))



        #TESTING
        print('**************** NOW TESTING ***********************')

        file = open('test_results.txt','w')
        error_num = 0
        for x, y, z, m in test_reader.iter():
            sentence_length = []
            for batch in m:
                sentence_length.append(sum(batch))
            loss, y_pred,train_index,test_index= session.run([
                train_model.class_loss,
                train_model.y_pred1,
                train_model.train_index,
                train_model.test_index
            ], {
                train_model.input_: x,
                train_model.class_targets: y,
                train_model.input_word: z,
                train_model.input_mask: m,
                train_model.sentence_length: sentence_length,
                train_model.dropout: 1
            })
            word_batch = z.flatten()

            # substitute the #@RTURL
            patternHT = '#[\w]+'
            patternUSR = '@[\w]+'
            patternURL = 'http|www\.|^com[^\w]'

            inputword = np.ndarray.flatten(z)
            y_predtrue = []
            for label, word in zip(y_pred, inputword):
                if re.match(patternHT, word_vocab._index2token[word]):
                    y_predtrue.append(
                        to_categorical([label_vocab._token2index['HT']], num_classes=args.num_classes)[0])
                elif re.match(patternURL, args.word_vocab._index2token[word]):
                    y_predtrue.append(
                        to_categorical([label_vocab._token2index['URL']], num_classes=args.num_classes)[0])
                elif re.match(patternUSR, args.word_vocab._index2token[word]):
                    y_predtrue.append(
                        to_categorical([label_vocab._token2index['USR']], num_classes=args.num_classes)[0])
                else:
                    label_index = np.argmax(label)
                    ht_index = label_vocab._token2index['HT']
                    url_index = label_vocab._token2index['URL']
                    usr_index = label_vocab._token2index['USR']
                    while (label_index == ht_index) or (label_index == url_index) or (label_index == usr_index):
                        label[label_index] = 0
                        label_index = np.argmax(label)
                    y_predtrue.append(label)
            y = np.reshape(y, (-1, args.num_classes))

            class_correct_prediction = np.equal(np.argmax(y_predtrue, 1), np.argmax(y, 1))

            for index,y_p,y,word in zip(class_correct_prediction,np.argmax(y_predtrue, 1),test_index,word_batch):
                if label_vocab._index2token[y] != '+':
                    a = label_vocab._index2token[y_p]
                    b = label_vocab._index2token[y]
                    c = word_vocab._index2token[word]
                    file.write(c + '\t' + b + '\t' + a)
                    file.write('\n')
                    if index==False:
                        error_num+=1


        file.close()

        print('the true accuracy of testing epoch is %f:' % ((2291-error_num) / 2291))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--choice', type=int, choices=[0,1], help='choose your decision: retraining, reproducing, or applying on your dataser')
    parser.add_argument('--rnn_size', type=int, help='size of LSTM internal state')
    parser.add_argument('--kernels', help='CNN kernel widths')
    parser.add_argument('--kernel_features', help='number of features in the CNN kernel')
    parser.add_argument('--adv_l', type=int, help='meta-parameter lambda in gradient reversal layer (GRL)')
    parser.add_argument('--char_embed_size', type=int, help='dimensionality of character embeddings')
    parser.add_argument('--word_embed_size', help='this hyper parameter can not be changed')
    parser.add_argument('--max_word_length', type=int, help='maximum word length')
    parser.add_argument('--param_init', type=float, help='initialize parameters at')
    parser.add_argument('--batch_size', type=int, help='number of sequences to train on in parallel')
    parser.add_argument('--max_epochs', type=int, help='number of full passes through the training data')
    args = parser.parse_args()
    if args.rnn_size:
        FLAGS.rnn_size = args.rnn_size
    if args.kernels:
        FLAGS.kernels = args.kernels
    if args.kernel_features:
        FLAGS.kernel_features = args.kernel_features
    if args.adv_l:
        FLAGS.adv_l = args.adv_l
    if args.char_embed_size:
        FLAGS.char_embed_size = args.char_embed_size
    if args.param_init:
        FLAGS.param_init = args.param_init
    if args.batch_size:
        FLAGS.batch = args.batch_size
    if args.max_epochs:
        FLAGS.max_epochs = args.max_epochs
    start = time.time()

    if args.choice==1:
        main()
    else:
        reproduce()


    wholetime = time.time() - start
    hour = int(wholetime) // 3600
    minute = int(wholetime) % 3600 //60
    sec = wholetime - 3600 * hour - 60 * minute
    print("whole time:\n")
    print("%d:%d:%f" % (hour, minute, sec))
