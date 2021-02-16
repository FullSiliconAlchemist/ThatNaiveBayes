import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import re

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter


# Task 1 method for plotting frequencies
def plot_frequencies(data):
    num_neg = 0
    num_pos = 0

    for row in data:
        data_class = row[0].split(" ", 1)[1].split(" ", 2)[0]
        # Count number of classes
        if data_class == 'neg':
            num_neg += 1
        elif data_class == 'pos':
            num_pos += 1

    # Distribution of positive and negative reviews
    objects = ('Positive', 'Negative')
    y_pos = np.arange(len(objects))
    performance = [num_pos, num_neg]

    # Styling
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Frequency')
    plt.title('Frequency of Negative and Positive Reviews')

    # Show plot results
    # plt.show()

# Import data using pandas
data = pd.read_csv("all_sentiment_shuffled.txt", sep="\t", header=None)
# Convert to numpy object for fast iteration
np_data = data.to_numpy()

# Plot class frequencies method
plot_frequencies(np_data)
print('Plotting complete')


# Multinomial Naive bayes classification
# P(Negative | W1 AND W2 AND W3 AND ... WN) = P(Negative) * P(W1 | Negative) * P(W2 | Negative)
# P(W1 | Negative) = (# of times W1 appears in training data of negative class) / (# of words)
# [
#   ('W1', 10),
#   ('W2', 12),
#   etc...
# ]

refactored_data = []

# Remove unnecessary columns and format into multi-dimensional array
for row in np_data:
    # Get class
    data_class = [row[0].split(" ", 1)[1].split(" ", 2)[0]]
    # Get data and remove all non-words, then split by spaces
    fields = row[0].split(" ", 1)[1].split(" ", 2)[2]
    data_fields = re.sub('[\W_]+[ ]', '', fields)
    data_class.append(fields)
    refactored_data.append(data_class)

# Put refactored data into a new dataframe object and define the column names for scikit
df = pd.DataFrame(refactored_data)
df.columns = ['CATEGORY', 'CONTENT']

# Filter stop words in the english language
vectorizer = CountVectorizer(stop_words='english')

# Create dictionary object with vocabulary and word frequency
all_features = vectorizer.fit_transform(df.CONTENT)

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(all_features, df.CATEGORY, test_size=0.8)
print('Training complete')

nb = MultinomialNB()
nb.fit(X_train, y_train)
correct = (y_test == nb.predict(X_test)).sum()
print(f'{correct/len(y_test)*100}% of reviews classified correctly')
# print(nb.predict(X_test))

# Base Decision Tree

# Optimized Decision Tree (Entropy measurements)

print('\nDone')
