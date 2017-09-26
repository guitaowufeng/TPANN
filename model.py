from __future__ import print_function
from __future__ import division


import tensorflow as tf
import numpy as np
import get_wordembed
from flip_gradient import flip_gradient
from tensorflow.python.ops import variable_scope


def rnn_decoder(decoder_inputs,
                initial_state,
                cell,
                num_unroll_steps = 39,
                loop_function=None,
                scope=None):

  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    encoder = decoder_inputs
    decoder_inputs = tf.concat([tf.zeros_like(decoder_inputs),decoder_inputs],1)
    for i in range(num_unroll_steps):
      if i > 0:
          variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(decoder_inputs, state)
      w_v = tf.Variable(tf.truncated_normal([500,500],stddev = 0.1))
      b_v = tf.Variable(tf.zeros(500)+0.1)
      output1 = tf.matmul(output,w_v)+b_v
      decoder_inputs = tf.concat([output1,encoder],1)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state

def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])
        return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b

def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]

    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in xrange(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output

def tdnn(input_, kernels, kernel_features, scope='TDNN'):
    '''

    :input:           input float tensor of shape [(batch_size*num_unroll_steps) x max_word_length x embed_size]
    :kernels:         array of kernel sizes
    :kernel_features: array of kernel feature sizes (parallel to kernels)
    '''
    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

    max_word_length = input_.get_shape()[1]
    embed_size = input_.get_shape()[-1]

    # input_: [batch_size*num_unroll_steps, 1, max_word_length, embed_size]
    # '1' above is not needed
    input_ = tf.expand_dims(input_,-1)# hjq,(input_,1) origin

    layers = []
    with tf.variable_scope(scope):
        #kernels=[1,2,3,4,5,6,7]  kernel_features=[50,100,150,200,200,200,200]
        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
            reduced_length = max_word_length - kernel_size + 1

            # [batch_size x max_word_length x embed_size x kernel_feature_size] origin
            conv = conv2d(input_, kernel_feature_size, kernel_size, embed_size, name="kernel_%d" % kernel_size)# hjq
            # conv=[batch_size x num_unroll_steps,reduced_length,1,kernel_feature_size] hjq
            pool = tf.nn.max_pool(tf.tanh(conv), [1, reduced_length,1, 1], [1, 1, 1, 1], 'VALID')#hjq,origin: [1,1,reduce_length,1]
            # pool=[batch_size x num_unroll_steps,1,1,kernel_feature_size]
            layers.append(tf.squeeze(pool))#origin:(pool,[1,2])
            #tf.squeeze()Removes dimensions 1 =>[num_kernels,batch_size x num_unroll_steps,feature_size]

        if len(kernels) > 1:
            #[batch_size x num_unroll_steps , num_kernels x features]
            output = tf.concat(layers,1)
        else:
            output = layers[0]

    return output


class Model():
    def __init__(self, args, infer=False):

        self.kernels= [int(a) for a in args.kernels.strip('[').strip(']').split(',')]
        self.kernel_features=[int(a) for a in args.kernel_features.strip('[').strip(']').split(',')]

        self.input_ = tf.placeholder(tf.int32, shape=[args.batch_size, args.num_unroll_steps, args.max_word_length], name="input")
        self.input_word = tf.placeholder(tf.int32,shape=[args.batch_size, args.num_unroll_steps],name="input_word")
        self.sentence_length = tf.placeholder(tf.int32, shape=[args.batch_size], name="sentence_length")
        self.input_mask = tf.placeholder(tf.float32, shape=[args.batch_size, args.num_unroll_steps], name="input_mask")
        self.class_targets = tf.placeholder(tf.float32, [args.batch_size, args.num_unroll_steps,args.num_classes], name='class_targets')
        self.domain_targets = tf.placeholder(tf.float32, [None, 2])
        self.dropout = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)

        ''' First, embed characters '''
        with tf.variable_scope('Embedding'):
            char_embedding_r = tf.get_variable('char_embedding', [args.char_vocab_size,args.char_embed_size])
            char_embeddinglist=tf.unstack(char_embedding_r)
            char_embeddinglist[0]=tf.zeros([args.char_embed_size],dtype=tf.float32)

            #get pre_train word_embedding
            word_in_pretrain = []
            wordembed_in_pretrain = []
            model = get_wordembed.load_model()
            word_index_list = args.word_vocab._token2index.items()
            b = 0
            # file = open('loss_word.txt', 'w')
            for item,_ in sorted(word_index_list, key= lambda  x : x[1]):
                word_in_pretrain.append(item)
            for item in word_in_pretrain:

                try:
                    wordembed_in_pretrain.append(model[item.lower().encode('utf8')])
                except KeyError:
                    #model embedding size

                    # file.write(item.lower().encode('utf8'))
                    # file.write('\n')

                    wordembed_in_pretrain.append(np.random.rand(200))
                    b+=1

        print('there are %d words not in model'%b)
        # file.close()
        # construct embedding matrix
        # check whether embedding can change
        self.word_embedding= tf.Variable(np.array(wordembed_in_pretrain),trainable=True,name='word_embedding',dtype=tf.float32)

        #input word embedding=[batch_size,num_unroll_steps,word_embedding]
        input_embedded_word=tf.nn.embedding_lookup(self.word_embedding,self.input_word)

        input_embedded_word=tf.nn.dropout(input_embedded_word,self.dropout)

        self.char_embedding=tf.stack(char_embeddinglist)
        # [batch_size x max_word_length, num_unroll_steps, char_embed_size]

        input_embedded = tf.nn.embedding_lookup(self.char_embedding, self.input_)

        #add noise
        if self.dropout!=1:
            input_embedded = input_embedded + np.random.randn(*input_embedded.get_shape())*0
        
        # input_embedded = tf.nn.dropout(input_embedded,self.dropout)
        # different from above '#', [batch_size x num_unroll_steps,max_word_length,char_embed_size]
        input_embedded_s = tf.reshape(input_embedded, [-1, args.max_word_length, args.char_embed_size])

        ''' Second, apply convolutions '''
        # [batch_size x num_unroll_steps, cnn_size]  # where cnn_size=sum(kernel_features)
        input_cnn = tdnn(input_embedded_s, self.kernels, self.kernel_features)

        ''' Maybe apply Highway '''
        #highway_layers=3
        if args.highway_layers > 0:
            input_cnn = highway(input_cnn, input_cnn.get_shape()[-1], num_layers=args.highway_layers)

        input_cnn = tf.reshape(input_cnn, [args.batch_size, args.num_unroll_steps, -1])
        #concat cnn_output and pre_train word_embedding
        input_cnn2= tf.concat([input_cnn,input_embedded_word], 2)

        with tf.variable_scope('LSTM'):
            # Foreward direction cell
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(args.rnn_size, forget_bias=1.0)
            # Backward direction cell
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(args.rnn_size, forget_bias=1.0)


            # self.initial_rnn_fwstate = lstm_fw_cell.zero_state(args.batch_size, dtype=tf.float32)
            # self.initial_rnn_bwstate = lstm_bw_cell.zero_state(args.batch_size, dtype=tf.float32)

            # outputs_size:[batch_size, max_time, output_size]
            outputs, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_cnn2, sequence_length=self.sentence_length, dtype=tf.float32)
            outputs = tf.concat(outputs,-1)

            if self.dropout != 1:
                outputs=tf.nn.dropout(outputs,keep_prob=self.dropout)#hjq

        with tf.variable_scope('autoencoder'):
            decoder_lstm_cell = tf.contrib.rnn.BasicLSTMCell(2 * args.rnn_size, forget_bias=1.0)
            decoder_lstm_cell = tf.contrib.rnn.MultiRNNCell([decoder_lstm_cell for i in range(3)])

            initial_cstate = tf.concat([state[0][0], state[1][0]], 1)
            initial_hstate = tf.concat([state[0][1], state[1][1]], 1)
            initial_state = [tf.contrib.rnn.LSTMStateTuple(initial_cstate, initial_hstate)] * 3


            # _,initial_hstate0 = tf.split(1,2,outputs[0])
            # initial_hstate1,_ = tf.split(1,2,outputs[-1])
            # initial_hstate = tf.concat(1,[initial_hstate0,initial_hstate1])
            # print('state shape',initial_state.get_shape())
            # print('cstate shape',initial_hstate.get_shape())


            # decoder_inputs = tf.zeros([args.batch_size, 2 * args.rnn_size])
            # decoder_inputs = decoder_inputs + tf.unpack(input_embedded_word,axis=1)
            # decoder_inputs = decoder_inputs[0:-1]

            decoder_labels = tf.one_hot(self.input_word, args.word_vocab_size)
            decoder_labels = tf.reshape(decoder_labels, [-1, args.word_vocab_size])

            # decoder_outputs [time,batch_size,hidden_size]
            decoder_outputs, decoder_output_state = rnn_decoder(initial_hstate, initial_state,
                                                                decoder_lstm_cell)

            decoder_outputs = tf.stack(decoder_outputs)  # [time,batch,2*hidden]
            decoder_outputs = tf.reshape(decoder_outputs, [-1, 2 * args.rnn_size])  # [time*batch,2*hidden]
            # decoder_W1 = tf.get_variable('decoder_W1',[2*args.decoder_rnn_size,600])
            # decoder_b1 = tf.get_variable('decoder_b1',[600])

            #    decoder_hidden = tf.tanh(tf.matmul(decoder_outputs,decoder_W1)+decoder_b1)

            decoder_W2 = tf.get_variable('decoder_W2', [2 * args.rnn_size, args.word_vocab_size])
            decoder_b2 = tf.get_variable('decoder_b2', [args.word_vocab_size])

            decoder_finish = tf.nn.softmax(
                tf.matmul(decoder_outputs, decoder_W2) + decoder_b2)  # [time,batch,vocb]
            # decoder_finish = tf.reshape(decoder_finish, [-1, args.batch_size,args.word_vocab_size])
            # encoder_input = tf.pack(input_cnn3)
            mask = tf.reshape(self.input_mask, [-1, 1])

            self.autoencoder_loss = 0.4 * tf.reduce_sum(-decoder_labels * tf.log(decoder_finish)*mask) / tf.reduce_sum(tf.cast(self.sentence_length,tf.float32))


                # MLP for class prediction
        with tf.variable_scope('label_predictor'):
            self.logits1=[]
            for idx, output in enumerate(tf.unstack(outputs, axis=0)):
                with tf.variable_scope('MLP') as scope:
                    if idx > 0:
                        scope.reuse_variables()
                    layer1 = tf.nn.softmax(linear(output, args.num_classes))
                    #w_c = tf.Variable(tf.truncated_normal([100,args.num_classes],stddev = 0.1))
                    #b_c = tf.Variable(tf.zeros(args.num_classes)+0.1)
                    #layer2 = tf.nn.softmax(tf.matmul(layer1,w_c)+b_c)
                # with tf.variable_scope('WordEmbedding2') as scope:
                #     if idx > 0:
                #         scope.reuse_variables()
                #     layer2 = tf.nn.softmax(linear2(layer1, args.num_classes))
                self.logits1.append(layer1)

            # y_pred1[batch_size, time, num_classes]
            self.y_pred1 = tf.reshape(self.logits1, [-1, args.num_classes])
            self.y1 = tf.reshape(self.class_targets, [-1, args.num_classes])

            mask = tf.reshape(self.input_mask,[-1,1])
            self.class_loss = -tf.reduce_sum(self.y1 * tf.log(self.y_pred1)*mask, name='loss') / tf.reduce_sum(tf.cast(self.sentence_length,tf.float32))



        self.train_index = tf.argmax(self.y_pred1,1)
        self.test_index = tf.argmax(self.y1,1)

        self.class_correct_prediction = tf.equal(tf.argmax(self.y_pred1, 1), tf.argmax(self.y1, 1))
        self.class_accuracy = tf.reduce_sum(tf.cast(self.class_correct_prediction,tf.float32)*tf.squeeze(mask))/tf.reduce_sum(mask)

        self.correct_num = tf.reduce_sum(tf.cast(self.class_correct_prediction,tf.float32)*tf.squeeze(mask))
        self.sum_num = tf.reduce_sum(mask)

        with tf.variable_scope('domain_predictor'):

            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(outputs, args.adv_l)

            self.logits2 = []
            for idx, output in enumerate(tf.unstack(feat, axis=0)):
                with tf.variable_scope('adversarial') as scope:
                    if idx > 0:
                        scope.reuse_variables()
                    layer1 = tf.nn.softmax(tf.nn.tanh(linear(output, 100)))
                    w_d = tf.Variable(tf.truncated_normal([100, 2], stddev=0.1))
                    b_d = tf.Variable(tf.zeros(2) + 0.1)
                    layer2 = tf.nn.softmax(tf.matmul(layer1, w_d) + b_d)
                # with tf.variable_scope('WordEmbedding4') as scope:
                #     if idx > 0:
                #         scope.reuse_variables()
                #     layer2 = tf.nn.softmax(linear2(layer1, 2))
                self.logits2.append(layer2)

            y_pred2 = tf.reshape(self.logits2, [-1, 2])
            y2 = self.domain_targets

            mask = tf.reshape(self.input_mask, [-1, 1])
            self.domain_loss = -tf.reduce_sum(y2 * tf.log(y_pred2) * mask, name='loss') / tf.reduce_sum(
                tf.cast(self.sentence_length, tf.float32))

            self.domain_correct_prediction = tf.equal(tf.argmax(y_pred2, 1), tf.argmax(y2, 1))
            self.domain_accuracy = tf.reduce_sum(
                tf.cast(self.domain_correct_prediction, tf.float32) * tf.squeeze(mask)) / tf.reduce_sum(mask)





        self.class_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.class_loss)

        self.total_loss = self.class_loss + self.domain_loss
        self.autoencoder_domain_cost = self.domain_loss + self.autoencoder_loss
        self.total_train_op = tf.train.AdagradOptimizer(0.1).minimize(self.total_loss)
        self.domain_train_op = tf.train.AdagradOptimizer(0.1).minimize(self.autoencoder_domain_cost)


