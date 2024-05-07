from copy import deepcopy
from alignment_methods import procrustes, closed_form_linear_regression, direct_alignment, weighted_lr, linear_regression
import numpy as np
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
import os


def w2v_to_numpy(model):
    """ Convert the word2vec model (the embeddings) into numpy arrays.
    Also create and return the mapping of words to the row numbers.

    Parameters:
    ===========
    model (gensim.Word2Vec): a trained gensim model

    Returns:
    ========
    embeddings (numpy.ndarray): Embeddings of each word
    idx, iidx (tuple): idx is a dictionary mapping word to row number
                        iidx is a dictionary mapping row number to word
    """
    model.wv.fill_norms()
    embeddings = deepcopy(model.wv.get_normed_vectors())
    idx = {w: i for i, w in enumerate(model.wv.index_to_key)}
    iidx = {i: w for i, w in enumerate(model.wv.index_to_key)}
    return embeddings, (idx, iidx)


def create_words_and_indices(data):
    english_words = []
    german_words = []
    english_indices = []
    german_indices = []

    for item in data:
        parts = item.split()
        english_words.append(parts[0])
        german_words.append(parts[1])
        english_indices.append(int(parts[2]))
        german_indices.append(int(parts[3]))
    return english_words, german_words, english_indices, german_indices

def load_data(data_path):
    with open(data_path, 'r') as f:
        data = f.readlines()
    return data


def sim_matrix(matrix, vector, sim_metric='cos'):
    if sim_metric == 'cos':
        vector= vector / np.linalg.norm(vector)
        matrix = matrix / np.linalg.norm(matrix, axis=1)[:, None]
        similarities = np.dot(matrix, vector)
    else:
        similarities = np.dot(matrix, vector)
    return similarities


def find_closest_words(matrix, vector, iidx, n, sim_metric='cos'):
    """
    Find the n closest words to a given vector in a matrix.

    Parameters:
    ===========
    matrix (numpy.ndarray): The matrix of embeddings.
    vector (numpy.ndarray): The vector to compare to.
    n (int): The number of closest words to find.

    Returns:
    ========
    list: The n closest words to the vector.
    """
    # Compute the cosine similarity between the vector and all the vectors in the matrix
    similarities = sim_matrix(matrix, vector, sim_metric)

    # Find the n closest words
    closest_indices = np.argsort(similarities)[::-1][:n]
    closest_words = [iidx[i] for i in closest_indices]

    return closest_words

def get_accuracy_scores(target_matrix, aligned_matrix, target_indices, aligned_indices, target_iidx, n):
    correct = 0
    for index, item in enumerate(aligned_indices):
        aligned_vector = aligned_matrix[item]
        closest_words = find_closest_words(target_matrix, aligned_vector, target_iidx, n)
        gold_word = target_iidx[target_indices[index]]
        # print(gold_word, closest_words)
        if gold_word in closest_words:
            correct += 1
    return correct / len(target_indices), len(target_indices)


def create_weight_matrix(en_indices, de_indices, type='freq'):
    zipped = list(zip(en_indices, de_indices))
    if type == 'freq':
        sum = [x + y for x, y in zipped]
        max_val = max(sum)
        min_val = min(sum)
        weight = [2 - (x - min_val) / (max_val - min_val) for x in sum]
    elif type == 'idiff':
        diff = [abs(x - y) for x, y in zipped]
        max_val = max(diff)
        min_val = min(diff)
        if max_val == min_val:
            weight = [1 for x in diff]
            return weight
        weight = [2 - (x - min_val) / (max_val - min_val) for x in diff]
    else:
        raise ValueError('Invalid weight type')
    return weight


def create_temp_train_set(train_data, type='random', size=50):
    if type == 'random':
        return np.random.choice(train_data, size=size, replace=False)

    if type == 'freq':
        train_ordered = sorted(train_data, key=lambda x: int(x.split()[2]) + int(x.split()[3]))
        return train_ordered[:size]
    elif type == 'idiff':
        train_ordered = sorted(train_data, key=lambda x: abs(int(x.split()[2]) - int(x.split()[3])))
        return train_ordered[:size]
    else:
        raise ValueError('Invalid type')







def experiment_setup(en_path, de_path, train_path, test_path, alignment_method='p', rank_type='freq', save_path=None, dataset='idiff',size=50):
    english_model = Word2Vec.load(en_path)
    german_model = Word2Vec.load(de_path)

    english_embeddings, (english_idx, english_iidx) = w2v_to_numpy(english_model)
    german_embeddings, (german_idx, german_iidx) = w2v_to_numpy(german_model)

    train_data = load_data(train_path)
    test_data = load_data(test_path)

    if dataset:
        train_data = create_temp_train_set(train_data, type=dataset, size=size)
    en_words, de_words, en_indices, de_indices = create_words_and_indices(train_data)
    en_test_words, de_test_words, en_test_indices, de_test_indices = create_words_and_indices(test_data)
    en_train_matrix = english_embeddings[en_indices]
    de_train_matrix = german_embeddings[de_indices]

    en_test_matrix = english_embeddings[en_test_indices]
    de_test_matrix = german_embeddings[de_test_indices]



    if alignment_method == 'p':
        p_matrix = procrustes(de_train_matrix, en_train_matrix)
    elif alignment_method == 'lr':
        p_matrix = closed_form_linear_regression(de_train_matrix, en_train_matrix)
    elif alignment_method == 'd':
        p_matrix = direct_alignment(de_train_matrix, en_train_matrix)
    elif alignment_method == 'w':
        train_weight = create_weight_matrix(en_indices, de_indices, type=rank_type)
        test_weight = create_weight_matrix(en_test_indices, de_test_indices, type=rank_type)
        print(de_train_matrix.shape, en_train_matrix.shape, de_test_matrix.shape, en_test_matrix.shape, len(train_weight), len(test_weight))
        model = weighted_lr(de_train_matrix, en_train_matrix, de_test_matrix, en_test_matrix, train_weight, test_weight)
    elif alignment_method == 'lr_t':
        model = linear_regression(de_train_matrix, en_train_matrix, de_test_matrix, en_test_matrix)


    else:
        raise ValueError('Invalid alignment method')

    if alignment_method != 'w' and alignment_method != 'lr_t':
        de_aligned = german_embeddings @ p_matrix
    else:
        de_aligned = model(torch.tensor(german_embeddings, dtype=torch.float32)).detach().numpy()

    save_items = []

    ## Training Accuracy
    accuracy, total = get_accuracy_scores(english_embeddings, de_aligned, en_indices, de_indices, english_iidx, 1)
    print('top 1 accuracy')
    print(f"Accuracy: {accuracy}, Total: {total}")
    save_items.append(accuracy)


    # top five accuracy
    accuracy, total_train = get_accuracy_scores(english_embeddings, de_aligned, en_indices, de_indices, english_iidx, 5)
    print('top 5 accuracy')
    print(f"Accuracy: {accuracy}, Total: {total}")
    save_items.append(accuracy)


    ### Testing Accuracy
    accuracy, total = get_accuracy_scores(english_embeddings, de_aligned, en_test_indices, de_test_indices, english_iidx, 1)
    print('top 1 accuracy')
    print(f"Accuracy: {accuracy}, Total: {total}")
    save_items.append(accuracy)

    # top five accuracy
    accuracy, total_test = get_accuracy_scores(english_embeddings, de_aligned, en_test_indices, de_test_indices, english_iidx, 5)
    print('top 5 accuracy')
    print(f"Accuracy: {accuracy}, Total: {total}")
    save_items.append(accuracy)


    # Save the results
    if save_path:
        if alignment_method == 'w':
            save_name = f'{alignment_method}_{rank_type}_{total_train}_{total_test}'
        else:
            save_name = f'{alignment_method}_{total_train}_{total_test}'

        if dataset:
            save_name += f'_{dataset}'

        save_name += '.txt'
        save_path = os.path.join(save_path, save_name)
        with open(save_path, 'w') as f:
            for item in save_items:
                f.write(f"{item}\n")

if __name__ == '__main__':
    en_path = 'data/english_model_lemmatized'
    de_path = 'data/german_model_lemmatized'
    train_path = 'train_test_data/train_set.txt'
    test_path = 'train_test_data/test_set.txt'

    alignment_method_list = ['lr_t', 'p', 'lr', 'd', 'w']

    # for alignment_method in alignment_method_list:
    #     experiment_setup(en_path, de_path, train_path, test_path, alignment_method=alignment_method, rank_type='idiff', save_path='results')
    #     if alignment_method == 'w':
    #         experiment_setup(en_path, de_path, train_path, test_path, alignment_method=alignment_method, rank_type='freq', save_path='results')
    #
    experiment_setup(en_path, de_path, train_path, test_path, alignment_method='lr_t', save_path='results', dataset=None)
    # numbers = [50, 100, 150]
    # datasets = ['random', 'freq', 'idiff']
    # for number in numbers:
    #     for data in datasets:
    #         experiment_setup(en_path, de_path, train_path, test_path, alignment_method='lr_t', rank_type='freq', save_path='results', dataset=data, size=number)