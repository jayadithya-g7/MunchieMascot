import os
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.saving import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
import multiprocessing

RECIPES_PATH = "recipes.csv"
GLOVE_PATH = "glove.6B.100d.txt"
MODEL_PATH = "model.keras"


def read_csv_with_errors(file_path):
    return pd.read_csv(file_path)


def write_csv(file_path, df: pd.DataFrame):
    df.to_csv(file_path)


def create_sequential_model():
    model = Sequential()

    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[
              embedding_matrix], trainable=False))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(intents_labels), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=3, batch_size=128,
                        validation_data=(X_test, y_test), verbose=1)

    model.save(MODEL_PATH)
    return model


def load_sequential_model():
    return load_model(MODEL_PATH)


recipes_df = read_csv_with_errors(RECIPES_PATH)


# recipes_df = pd.DataFrame(recipes_data[1:], columns=recipes_data[0])
if not recipes_df.columns.isin(['corpus']).any():
    recipes_df['steps'] = recipes_df['steps'].fillna('').str.lower()
    recipes_df['ingredients'] = recipes_df['ingredients'].fillna(
        '').str.lower()
    recipes_df['corpus'] = recipes_df['steps'] + \
        ' ' + recipes_df['ingredients']
    # write_csv(RECIPES_PATH, recipes_df)

intents = {
    "greeting": ["hi", "hello", "hey", "howdy"],
    "goodbye": ["bye", "goodbye", "see you later", "take care"],
    "thanks": ["thanks", "thank you", "appreciate it"],
    "recipe_search": ["suggest a recipe with", "recipe with", "recipe for"],
    "ingredient_substitution": ["substitute for", "replace", "alternative for"],
    "cooking_time": ["how long", "take to cook", "cooking time"],
    "nutritional_information": ["nutritional value", "calories in", "nutrients in"],
    "recipe_modification": ["make without", "substitute", "alternative"],
    "allergy_friendly_recipe": ["gluten-free", "dairy-free", "nut-free"],
    "cooking_technique": ["how to", "properly", "technique for"],
    "current_day": ["what day is it", "day of the week"],
    "weather": ["weather forecast", "current weather"],
    "bot_name": ["munchiebot"]
}


tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(recipes_df['corpus'])
vocab_size = len(tokenizer.word_index) + 1


max_len = max([len(seq.split()) for seq in recipes_df['corpus']])
sequences = tokenizer.texts_to_sequences(recipes_df['corpus'])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')


intents_labels = {
    "greeting": 0,
    "goodbye": 1,
    "thanks": 2,
    "recipe_search": 3,
    "ingredient_substitution": 4,
    "cooking_time": 5,
    "nutritional_information": 6,
    "recipe_modification": 7,
    "allergy_friendly_recipe": 8,
    "cooking_technique": 9,
    "current_day": 10,
    "weather": 11,
    "bot_name": 12
}


labels = []
for text in recipes_df['corpus']:
    label = -1
    for intent, keywords in intents.items():
        if any(keyword in text for keyword in keywords):
            label = intents_labels[intent]
            break
    labels.append(label)


labels = to_categorical(labels, num_classes=len(intents_labels))


X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42)

print(len(X_train))

embeddings_index = {}
with open(GLOVE_PATH, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = None
if os.path.exists(MODEL_PATH):
    model = load_sequential_model()
else:
    model = create_sequential_model()

# input_sequences = tokenizer.texts_to_sequences(["recipe", "for", "soup"])
# print(tokenizer.sequences_to_texts(input_sequences))
# input_padded = pad_sequences(input_sequences, maxlen=max_len, padding='post')
# predictions = model.predict(input_padded)
# print(predictions)
# predicted_classes = np.argmax(predictions, axis=1)
# print(predicted_classes)
# predicted_texts = tokenizer.sequences_to_texts(
#     sequences=[[idx] for idx in predicted_classes])
# print(predicted_texts)

input_text = "suggest a recipe with chicken"
input_sequences = tokenizer.texts_to_sequences([input_text.split(' ')])
input_padded = pad_sequences(input_sequences, maxlen=max_len, padding='post')
prediction = model.predict(input_padded)[0]
print(prediction)
predicted_class_idx = np.argmax(prediction)
intent_labels = {v: k for k, v in intents_labels.items()}
predicted_intent = intent_labels[predicted_class_idx]

print(f"Predicted intent: {predicted_intent}")
