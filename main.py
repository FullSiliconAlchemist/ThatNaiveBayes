import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import re

data = pd.read_csv("all_sentiment_shuffled.txt", sep="\t")

# TASK 1
num_neg = 0
num_pos = 0

for index, row in data.iterrows():
    data_class = [row[0].split(" ", 1)[1].split(" ", 2)[0]]
    # Count number of classes
    if data_class[0] == 'neg':
        num_neg += 1
    elif data_class[0] == 'pos':
        num_pos += 1

# Split data into two groups, one for training one for testing
training = math.floor(data.shape[0] * 0.8)

# Slicing of different data groups
training_data = data.iloc[:training]
testing_data = data.iloc[training:]

training_data_arr = []

# Remove unnecessary columns and format into multi-dimensional array
for index, row in training_data.iterrows():
    # Get class
    data_class = [row[0].split(" ", 1)[1].split(" ", 2)[0]]
    # Get data and remove all non-words, then split by spaces
    fields = row[0].split(" ", 1)[1].split(" ", 2)[2]
    data_fields = re.sub('[\W_]+[ ]', '', fields).split(" ")
    data_class.append(data_fields)
    training_data_arr.append(data_class)

print('Data cleaning complete')

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
plt.show()

# Naive bayes classification


# How it works:
# P(Negative | W1 AND W2 AND W3 AND ... WN) = P(Negative) * P(W1 | Negative) * P(W2 | Negative)

print('\nDone')

