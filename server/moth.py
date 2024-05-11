import re
import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.saving import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from transformers import TFBertModel, BertTokenizer
from sklearn.model_selection import train_test_split
import os


RECIPES_PATH = "recipes.csv"
GLOVE_PATH = "glove.6B.100d.txt"
MODEL_PATH = "moth-model.keras"


def read_csv_with_errors(file_path):
    return pd.read_csv(file_path)


def write_csv(file_path, df: pd.DataFrame):
    df.to_csv(file_path)


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
recipes_df = read_csv_with_errors(RECIPES_PATH)


# recipes_df = pd.DataFrame(recipes_data[1:], columns=recipes_data[0])
if not recipes_df.columns.isin(['corpus']).any():
    recipes_df['steps'] = recipes_df['steps'].fillna('').str.lower()
    recipes_df['ingredients'] = recipes_df['ingredients'].fillna(
        '').str.lower()
    recipes_df['corpus'] = recipes_df['steps'] + \
        ' ' + recipes_df['ingredients']
# Load pre-trained BERT model and tokenizer
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a function to preprocess and encode input text


def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    tokens = bert_tokenizer.encode_plus(
        text, add_special_tokens=True, return_tensors='tf', padding='max_length', truncation=True)
    return tokens['input_ids'], tokens['attention_mask']

# Create a new model architecture


def load_sequential_model():
    return load_model(MODEL_PATH)


def create_sequential_model():
    model = Sequential()
    model.add(bert_model)
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    model.add(Dense(len(intents_labels), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(recipes_df['corpus'])
vocab_size = len(tokenizer.word_index) + 1


max_len = max([len(seq.split()) for seq in recipes_df['corpus']])
sequences = tokenizer.texts_to_sequences(recipes_df['corpus'])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Preprocess the training data
X_train_input_ids, X_train_attention_masks = [], []
for text in recipes_df['corpus']:
    input_ids, attention_mask = preprocess_text(text)
    X_train_input_ids.append(input_ids)
    X_train_attention_masks.append(attention_mask)


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
# Function to get the intent from the model


def get_intent(input_text):
    input_ids, attention_mask = preprocess_text(input_text)
    prediction = model.predict(
        [np.array([input_ids]), np.array([attention_mask])])
    predicted_class_idx = np.argmax(prediction)
    predicted_intent = list(intents_labels.keys())[predicted_class_idx]
    return predicted_intent


# Example usage
user_input = "I want a recipe for soup"
intent = get_intent(user_input)
print(f"Predicted intent: {intent}")
