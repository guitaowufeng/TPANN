from gensim.models import word2vec
import logging



# main program
def load_model():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec.load('./word2vec/word2vec_200dim.model')
    print('the word train vector is:',model['train'])
    return model






# model_2 = word2vec.Word2Vec.load("text8.model")


# model.save_word2vec_format(u"tweet.model.bin", binary=True)
# model_3 = word2vec.Word2Vec.load_word2vec_format("text8.model.bin", binary=True)

if __name__ == "__main__":
    print('loading completed')
    pass
