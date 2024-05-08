from copy import deepcopy
from gensim.models import Word2Vec
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim





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


def procrustes(A, B):
    """
    Solve the orthogonal Procrustes problem which finds the matrix R that
    best maps matrix A onto matrix B under orthogonal transformation.

    Parameters:
    A (numpy.ndarray): The source matrix.
    B (numpy.ndarray): The target matrix to map A onto.

    Returns:
    numpy.ndarray: The orthogonal matrix R.
    """
    # Compute the matrix product of A^T and B
    M = A.T @ B

    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(M)

    # Compute R as U * V^T
    R = U @ Vt

    return R


def closed_form_linear_regression(A, B):
    """
    Perform closed form linear regression to find matrix X such that AX ≈ B.

    Args:
    A (np.array): A matrix of shape (m, n).
    B (np.array): B matrix of shape (m, p).

    Returns:
    X (np.array): The matrix X that approximates the solution to AX ≈ B.
    """
    # Compute A transpose
    A_T = A.T

    # Compute (A^T * A)
    A_T_A = A_T @ A

    # Compute the inverse of (A^T * A)
    A_T_A_inv = np.linalg.inv(A_T_A)

    # Compute (A^T * B)
    A_T_B = A_T @ B

    # Compute X = (A^T * A)^(-1) * (A^T * B)
    X = A_T_A_inv @ A_T_B

    return X

def direct_alignment(A, B):
    """
    Perform direct alignment to find matrix X such that AX ≈ B.

    Args:
    A (np.array): A matrix of shape (m, n).
    B (np.array): B matrix of shape (m, p).

    Returns:
    X (np.array): The matrix X that approximates the solution to AX ≈ B.
    """
    # Compute the matrix product of A^T and B
    X = np.linalg.pinv(A) @ B

    return X


# def linear_regression(A, B, A_test, B_test):
#     X = torch.tensor(A, dtype=torch.float32)
#     Y = torch.tensor(B, dtype=torch.float32)
#     X_test = torch.tensor(A_test, dtype=torch.float32)
#     Y_test = torch.tensor(B_test, dtype=torch.float32)
#
#     model = nn.Linear(X.shape[1], Y.shape[1], bias=False)
#     optimizer = optim.SGD(model.parameters(), lr=0.001)
#     patience = 500
#     best_loss = np.inf
#     epochs_no_improve = 0
#
#     for epoch in range(100000):
#         model.train()
#         optimizer.zero_grad()
#
#         predictions = model(X)
#         loss = nn.MSELoss()(predictions, Y)
#
#         loss.backward()
#         optimizer.step()
#
#         model.eval()
#         with torch.no_grad():
#             predictions = model(X_test)
#             loss_val = nn.MSELoss()(predictions, Y_test)
#
#         if loss_val < best_loss:
#             best_loss = loss_val
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve == patience:
#                 break
#
#     return model

def linear_regression(A, B, A_test, B_test):
    # Ensure all data are in the correct type and potentially the correct device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(A, dtype=torch.float32).to(device)
    Y = torch.tensor(B, dtype=torch.float32).to(device)
    X_test = torch.tensor(A_test, dtype=torch.float32).to(device)
    Y_test = torch.tensor(B_test, dtype=torch.float32).to(device)

    # Define the model
    model = nn.Linear(X.shape[1], Y.shape[1], bias=False).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    patience = 500
    best_loss = np.inf
    epochs_no_improve = 0

    # Training loop
    for epoch in range(1000000):
        model.train()
        optimizer.zero_grad()

        predictions = model(X)
        loss = criterion(predictions, Y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_test)
            loss_val = criterion(val_predictions, Y_test)

        # Early stopping logic
        if loss_val < best_loss - 1e-5:  # Adding a small threshold for significant improvement
            best_loss = loss_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Stopping early at epoch {epoch}')
                break

        # Optionally print loss information every 1000 epochs
        if epoch % 10000 == 0:
            print(f'Epoch {epoch}: Training Loss: {loss.item()}, Validation Loss: {loss_val.item()}')

    return model
def weighted_lr(A, B, A_test, B_test, weights_train, weights_test):
    # Data
    X = torch.tensor(A, dtype=torch.float32)
    Y = torch.tensor(B, dtype=torch.float32)
    X_test = torch.tensor(A_test, dtype=torch.float32)
    Y_test = torch.tensor(B_test, dtype=torch.float32)
    weights = torch.tensor(weights_train, dtype=torch.float32).view(-1, 1)
    weights_test = torch.tensor(weights_test, dtype=torch.float32).view(-1, 1)

    # Model
    model = nn.Linear(X.shape[1], Y.shape[1], bias=False)
    # Loss Function
    def weighted_mse_loss(input, target, weight):
        return torch.sum(weight * (input - target) ** 2)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    patience = 500
    best_loss = np.inf
    epochs_no_improve = 0

    # Training loop
    for epoch in range(100000):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X)

        # Compute weighted loss
        loss = weighted_mse_loss(predictions, Y, weights)

        # Backward pass
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            predictions = model(X_test)
            loss_val = weighted_mse_loss(predictions, Y_test, weights_test)

        if loss_val < best_loss:
            best_loss = loss_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break


    return model

def weighted_lr_train_only(A, B, weights):
    # Data
    X = torch.tensor(A, dtype=torch.float32)
    Y = torch.tensor(B, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32).view(-1, 1)

    # Model
    model = nn.Linear(X.shape[1], Y.shape[1], bias=False)
    # Loss Function
    def weighted_mse_loss(input, target, weight):
        return torch.sum(weight * (input - target) ** 2)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    patience = 500
    best_loss = np.inf
    epochs_no_improve = 0

    # Training loop
    for epoch in range(100000):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X)

        # Compute weighted loss
        loss = weighted_mse_loss(predictions, Y, weights)

        # Backward pass
        loss.backward()
        optimizer.step()

        if loss < best_loss:
            best_loss = loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break

    return model

# returns the words and indicies given a dataset from train_test_data
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


def create_matrix_slice(matrix, indices):

    return matrix[indices]

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
    similarities = sim_matrix(matrix, vector, sim_metric='cos')

    # Find the n closest words
    closest_indices = np.argsort(similarities)[::-1][:n]
    closest_words = [iidx[i] for i in closest_indices]

    return closest_words

def sim_matrix(matrix, vector, sim_metric='cos'):
    if sim_metric == 'cos':
        vector= vector / np.linalg.norm(vector)
        matrix = matrix / np.linalg.norm(matrix, axis=1)[:, None]
        similarities = np.dot(matrix, vector)
    else:
        similarities = np.dot(matrix, vector)
    return similarities

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



# top_1_count = 0
# total_count = 0
# for i, j in enumerate(de_indices):
#     aligned_vector = de_english_aligned[j]
#     closest_word = find_closest_words(aligned_vector, en_emb, en_item[1], 1)[0]
#     english_index = en_indices[i]
#     gold_word = en_item[1][english_index]
#     if closest_word == gold_word:
#         print(f'Correctly aligned {gold_word} to {closest_word}')
#         top_1_count += 1
#     else:
#         print(f'Incorrectly aligned {gold_word} to {closest_word}')
#     total_count += 1
# print(f'Top 1 accuracy: {top_1_count/total_count}')
# print(f'Total count: {total_count}')

if __name__ == "__main__":
    english_model = Word2Vec.load(os.path.join('data', 'english_model_lemmatized'))
    german_model = Word2Vec.load(os.path.join('data', 'german_model_lemmatized'))

    english_embeddings, (english_idx, english_iidx) = w2v_to_numpy(english_model)
    german_embeddings, (german_idx, german_iidx) = w2v_to_numpy(german_model)

    train_data = open(os.path.join('train_test_data', 'train_set.txt'), 'r').read().splitlines()
    test_data = open(os.path.join('train_test_data', 'test_set.txt'), 'r').read().splitlines()

    en_words, de_words, en_indices, de_indices = create_words_and_indices(train_data)
    en_test_words, de_test_words, en_test_indices, de_test_indices = create_words_and_indices(test_data)

    en_train_matrix = create_matrix_slice(english_embeddings, en_indices)
    de_train_matrix = create_matrix_slice(german_embeddings, de_indices)

    p_matrix = procrustes(de_train_matrix, en_train_matrix)

    de_aligned = german_embeddings @ p_matrix


    ## Training Accuracy
    accuracy, total = get_accuracy_scores(english_embeddings, de_aligned, en_indices, de_indices, english_iidx, 1)
    print('top 1 accuracy')
    print(f"Accuracy: {accuracy}, Total: {total}")

    # top five accuracy
    accuracy, total = get_accuracy_scores(english_embeddings, de_aligned, en_indices, de_indices, english_iidx, 5)
    print('top 5 accuracy')
    print(f"Accuracy: {accuracy}, Total: {total}")

    ### Testing Accuracy
    accuracy, total = get_accuracy_scores(english_embeddings, de_aligned, en_test_indices, de_test_indices, english_iidx, 1)
    print('top 1 accuracy')
    print(f"Accuracy: {accuracy}, Total: {total}")

    # top five accuracy
    accuracy, total = get_accuracy_scores(english_embeddings, de_aligned, en_test_indices, de_test_indices, english_iidx, 5)
    print('top 5 accuracy')
    print(f"Accuracy: {accuracy}, Total: {total}")


    ### Closed Form Linear Regression
    X = closed_form_linear_regression(de_train_matrix, en_train_matrix)
    de_aligned = german_embeddings @ X

    print('Closed Form Linear Regression')

    ## Training Accuracy
    accuracy, total = get_accuracy_scores(english_embeddings, de_aligned, en_indices, de_indices, english_iidx, 1)
    print('top 1 accuracy')
    print(f"Accuracy: {accuracy}, Total: {total}")

    # top five accuracy
    accuracy, total = get_accuracy_scores(english_embeddings, de_aligned, en_indices, de_indices, english_iidx, 5)
    print('top 5 accuracy')
    print(f"Accuracy: {accuracy}, Total: {total}")

    ### Testing Accuracy
    accuracy, total = get_accuracy_scores(english_embeddings, de_aligned, en_test_indices, de_test_indices, english_iidx, 1)
    print('top 1 accuracy')
    print(f"Accuracy: {accuracy}, Total: {total}")

    # top five accuracy
    accuracy, total = get_accuracy_scores(english_embeddings, de_aligned, en_test_indices, de_test_indices, english_iidx, 5)
    print('top 5 accuracy')
    print(f"Accuracy: {accuracy}, Total: {total}")


    ### Direct Alignment
    X = direct_alignment(de_train_matrix, en_train_matrix)
    de_aligned = german_embeddings @ X

    print('Direct Alignment')

    ## Training Accuracy
    accuracy, total = get_accuracy_scores(english_embeddings, de_aligned, en_indices, de_indices, english_iidx, 1)
    print('top 1 accuracy')
    print(f"Accuracy: {accuracy}, Total: {total}")

    # top five accuracy
    accuracy, total = get_accuracy_scores(english_embeddings, de_aligned, en_indices, de_indices, english_iidx, 5)
    print('top 5 accuracy')
    print(f"Accuracy: {accuracy}, Total: {total}")

    ### Testing Accuracy
    accuracy, total = get_accuracy_scores(english_embeddings, de_aligned, en_test_indices, de_test_indices, english_iidx, 1)
    print('top 1 accuracy')
    print(f"Accuracy: {accuracy}, Total: {total}")

    # top five accuracy
    accuracy, total = get_accuracy_scores(english_embeddings, de_aligned, en_test_indices, de_test_indices, english_iidx, 5)
    print('top 5 accuracy')
    print(f"Accuracy: {accuracy}, Total: {total}")











