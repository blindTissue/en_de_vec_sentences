from gensim.models import Word2Vec

def create_word2vec(sentences, vector_size, window, min_count):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count)
    return model


def create_and_save_word2vec(sentences, vector_size, window, min_count, save_path):
    model = create_word2vec(sentences, vector_size, window, min_count)
    model.save(save_path)
    return model