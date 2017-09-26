import numpy as np
import os
import pickle
import codecs
import collections
from keras.utils.np_utils import to_categorical

class Vocab:
    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)


    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)


def load_data(file_name,dataname,max_sentnece_length,max_word_length):
    char_vocab = Vocab()
    char_vocab.feed(' ')
    word_vocab = Vocab()
    word_vocab.feed(' ')
    label_vocab = Vocab()
    label_vocab.feed('+')

    word_tokens = collections.defaultdict(list)
    char_tokens = collections.defaultdict(list)
    label_tokens = collections.defaultdict(list)

    input_word = collections.defaultdict(list)
    input_char = collections.defaultdict(list)
    input_label = collections.defaultdict(list)
    mask_label = collections.defaultdict(list)

    for data_name in (dataname):
        print('reading data:', data_name)

        file = codecs.open(os.path.join(file_name,data_name), 'r', 'utf-8')

        sentence = []#each sentence padding to max length
        word_char = []#padding word conrespanding to ' '
        sen_label = []#each sentence label padding to max length

        mask = []

        a=0
        for line in file:
            line = line.strip()

            if line!='':
                onesplit=line.split('\t')[0]
                labelsplit=line.split('\t')[1]

            # for word,label in zip(onesplit,labelsplit):
                word_tokens[data_name].append(word_vocab.feed(onesplit))
                mask.append(1)
                if len(onesplit) > max_word_length:
                    char_array = [char_vocab.feed(c) for c in onesplit[:max_word_length]]
                else:
                    char_array = [char_vocab.feed(c) for c in onesplit]
                char_tokens[data_name].append(char_array)
                label_tokens[data_name].append(label_vocab.feed(labelsplit))


                sentence.append(word_vocab.feed(onesplit))
                word_char.append(char_array)
                sen_label.append(label_vocab.feed(labelsplit))
            else:
                a+=1
                if len(sentence) < max_sentnece_length:#padding
                    for i in range(max_sentnece_length - len(sentence)):
                        sentence.append(word_vocab.feed(' '))
                        mask.append(0)
                        sen_label.append(label_vocab.feed('+'))
                        word_char.append([char_vocab.feed(' ')])
                if len(sentence) > max_sentnece_length:#cutting
                    sentence = sentence[:max_sentnece_length]
                    mask = mask[:max_sentnece_length]
                    sen_label = sen_label[:max_sentnece_length]
                    word_char = word_char[:max_sentnece_length]

                input_word[data_name].append(sentence)
                input_char[data_name].append(word_char)
                input_label[data_name].append(sen_label)
                mask_label[data_name].append(mask)
                if len(sentence)!=max_sentnece_length:
                    print(sentence)


                sentence = []
                sen_label = []
                word_char = []
                mask = []


##############################################################################
    word_tensors = {}
    char_tensors = {}
    label_tensors = {}
    mask_tensors = {}
    for fname in (dataname):
        assert len(input_word[fname]) == len(input_char[fname])
        assert len(input_word[fname]) == len(input_label[fname])
        print('tranforming numpy array:',fname)

        word_tensors[fname] = np.array(input_word[fname],dtype=np.int32)
        char_tensors[fname] = np.zeros([len(input_char[fname]), max_sentnece_length, max_word_length], dtype=np.int32)
        label_tensors[fname] = np.array(input_label[fname], dtype=np.int32)
        mask_tensors[fname] = np.array(mask_label[fname], dtype=np.int32)

        for i, word_array in enumerate(input_char[fname]):
            for j,char_array in enumerate(word_array):
                char_tensors[fname][i,j, :len(char_array)] = np.array(char_array,dtype=np.int32)

    print('label vocab size:',label_vocab.size)
    print('label vocab index to token:',label_vocab._index2token)

    return word_vocab, char_vocab, label_vocab, word_tensors, char_tensors, label_tensors,mask_tensors






class DataReader:
    def __init__(self, word_tensor, char_tensor,label_tensor, mask_tensor, batch_size, num_class):
        length = word_tensor.shape[0]
        assert char_tensor.shape[0] == length

        max_sentence_length = char_tensor.shape[1]
        max_word_length = char_tensor.shape[2]

        #for better reshape
        reduced_length = (length // batch_size) * batch_size
        word_tensor1 = word_tensor[:reduced_length,:]
        self.word_tensor2 = word_tensor[reduced_length:, :]
        char_tensor1 = char_tensor[:reduced_length, :,:]
        self.char_tensor2 = char_tensor[reduced_length:, :, :]
        mask_tensor1 = mask_tensor[:reduced_length, :]
        self.mask_tensor2 = mask_tensor[reduced_length:, :]

        label_tensor = np.reshape(label_tensor,[-1])
        label_tensor = to_categorical(label_tensor,num_classes=num_class)
        label_tensor = np.reshape(label_tensor,[length,max_sentence_length,num_class])

        label_tensor1 = label_tensor[:reduced_length,:,:]
        self.label_tensor2 = label_tensor[reduced_length:,:,:]


        w_batches = word_tensor1.reshape([-1, batch_size,max_sentence_length])
        x_batches = char_tensor1.reshape([-1, batch_size, max_sentence_length, max_word_length])
        y_batches = label_tensor1.reshape([-1, batch_size, max_sentence_length,num_class])
        m_batches = mask_tensor1.reshape([-1, batch_size,max_sentence_length])


        self._w_batches = list(w_batches)
        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        self._m_batches = list(m_batches)
        assert len(self._x_batches) == len(self._y_batches)==len(self._w_batches)==len(self._m_batches)
        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.num_unroll_steps = max_sentence_length

    def iter(self):
        for x, y, z, m in zip(self._x_batches, self._y_batches, self._w_batches, self._m_batches):
            yield x, y, z, m

if __name__ == '__main__':
    word_vocab, char_vocab, label_vocab, word_tensors, char_tensors, label_tensors,mask_tensors = load_data(file_name='../',max_sentnece_length=39,max_word_length=35)
    a = DataReader(word_tensors['domain_ptb'], char_tensors['domain_ptb'],label_tensors['domain_ptb'],mask_tensors['domain_ptb'], 20, 54)
    for c,d,e,m in a.iter():
        print('cccccccccccccccccccccccccccc',c)
        print('dddddddddddddddddddddddddddd',d)
        print('eeeeeeeeeeeeeeeeeeeeeeeeeeee',e)
        print('mmmmmmmmmmmmmmmmmmmmmmmmmmmm',m)
        assert 1==2


