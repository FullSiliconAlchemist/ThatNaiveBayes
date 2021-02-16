import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import re

from sklearn.naive_bayes import MultinomialNB
from collections import Counter


# Task 1 method for plotting frequencies
def plot_frequencies(data_pos, data_neg):
    # Distribution of positive and negative reviews
    objects = ('Positive', 'Negative')
    y_pos = np.arange(len(objects))
    performance = [data_pos, data_neg]

    # Styling
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Frequency')
    plt.title('Frequency of Negative and Positive Reviews')

    # Show plot results
    # plt.show()


data = pd.read_csv("all_sentiment_shuffled.txt", sep="\t")

np_data = data.to_numpy()

num_neg = 0
num_pos = 0

# Best not to iterate over dataframes, should convert large datasets to vectors
for x in range(np_data.shape[0]):
    data_class = np_data[x][0].split(" ", 1)[1].split(" ", 2)[0]
    # Count number of classes
    if data_class == 'neg':
        num_neg += 1
    elif data_class == 'pos':
        num_pos += 1

# Split data into two groups, one for training one for testing
training = math.floor(np_data.shape[0] * 0.8)

training_data_v1 = []
training_data_v2 = [[], []]

# Remove unnecessary columns and format into multi-dimensional array
for x in range(training):
    # Get class
    data_class = [np_data[x][0].split(" ", 1)[1].split(" ", 2)[0]]
    # Get data and remove all non-words, then split by spaces
    fields = np_data[x][0].split(" ", 1)[1].split(" ", 2)[2]
    data_fields = re.sub('[\W_]+[ ]', '', fields).split(" ")
    data_class.append(data_fields)
    training_data_v1.append(data_class)
    # V2 training
    if data_class[0] == 'pos':
        training_data_v2[0].append(data_fields)
    if data_class[0] == 'neg':
        training_data_v2[1].append(data_fields)


print('Data cleaning complete')

plot_frequencies(num_pos, num_neg)

# Multinomial Naive bayes classification
# P(Negative | W1 AND W2 AND W3 AND ... WN) = P(Negative) * P(W1 | Negative) * P(W2 | Negative)
# P(W1 | Negative) = (# of times W1 appears in training data of negative class) / (# of words)
# [
#   ('W1', 10),
#   ('W2', 12),
#   etc...
# ]
neg_freq = Counter()
pos_freq = Counter()
tot_words = Counter()

for data in training_data_v1:
    # Get all words in vocab
    for w in data[1]:
        tot_words[w] += 1
    # Get all occurrences of negative words
    if data[0] == 'neg':
        for w in data[1]:
            neg_freq[w] += 1
    # Get all occurrences of positive words
    if data[0] == 'pos':
        for w in data[1]:
            pos_freq[w] += 1

tot_vocab = len(tot_words.values())

nb = MultinomialNB()
np_arr = np.array(training_data_v2)
nb.fit(np_arr, ['pos', 'neg'])

# print(training_data_v1)

# Base Decision Tree

# Optimized Decision Tree (Entropy measurements)

print('\nDone')
