import numpy as np
import pandas as pd
import string
from flask import Flask, render_template, request

app = Flask(__name__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string

# Read Data
spam_df = pd.read_csv('spam.csv', encoding="ISO-8859-1")
spam_df = spam_df[['v1', 'v2']]
spam_df.rename(columns={'v1': 'spam', 'v2': 'text'}, inplace=True)
spam_df.spam = spam_df.spam.apply(lambda s: True if s=='spam' else False)
spam_df.text = spam_df.text.apply(lambda t: t.lower().translate(str.maketrans('', '', string.punctuation)))
spam_df = spam_df.sample(frac=1)

# Split into Training and Testing Sets
train_spam_df = spam_df.iloc[:int(len(spam_df)*0.7)]
test_spam_df = spam_df.iloc[int(len(spam_df)*0.7):]
FRAC_SPAM_TEXTS = train_spam_df.spam.mean()


# Create Spam Bag of Words and Non-Spam Bag of Words
train_spam_words = ' '.join(train_spam_df[train_spam_df.spam == True].text).split(' ')
train_non_spam_words = ' '.join(train_spam_df[train_spam_df.spam == False].text).split(' ')
common_words = set(train_spam_words).intersection(set(train_non_spam_words))

train_spam_bow = dict()
for w in common_words:
    train_spam_bow[w] = train_spam_words.count(w) / len(train_spam_words)

train_non_spam_bow = dict()
for w in common_words:
    train_non_spam_bow[w] = train_non_spam_words.count(w) / len(train_non_spam_words)

# Predict on Test Set
def predict_text(t, verbose=False):
    valid_words = [w for w in t if w in train_spam_bow]
    spam_probs = [train_spam_bow[w] for w in valid_words]
    non_spam_probs = [train_non_spam_bow[w] for w in valid_words]
    if verbose:
        data_df = pd.DataFrame()
        data_df['word'] = valid_words
        data_df['spam_prob'] = spam_probs
        data_df['non_spam_prob'] = non_spam_probs
        data_df['ratio'] = [s/n if n > 0 else np.inf for s,n in zip(spam_probs, non_spam_probs)]
        print(data_df)
    spam_score = sum([np.log(p) for p in spam_probs]) + np.log(FRAC_SPAM_TEXTS)
    non_spam_score = sum([np.log(p) for p in non_spam_probs]) + np.log(1-FRAC_SPAM_TEXTS)
    if verbose:
        print('Spam Score: %s'%spam_score)
        print('Non-Spam Score: %s'%non_spam_score)
    return (spam_score >= non_spam_score)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['message']
    prediction = predict_text(user_input.lower().translate(str.maketrans('', '', string.punctuation)).split())
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
