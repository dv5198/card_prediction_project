import os
import random

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.db import transaction

from .models import Draw
from .forms import DrawForm

from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Reshape, BatchNormalization,
                                     Bidirectional)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.utils import to_categorical

def predict_view(request):
    return render(request, "predictions.html")


# --- Add a draw to the database ---
def add_draw(request):
    """
    Handles adding a new draw to the database.
    """
    if request.method == "POST":
        form = DrawForm(request.POST)
        print(form)
        if form.is_valid():
            Draw.objects.create(
                Club=row['Club'],
                Diamond=row['Diamond'],
                Heart=row['Heart'],
                spade=row['spade'],
            )
        return redirect("home")
    else:
        form = DrawForm()
        return render(request, "add_draw.html", {"form": form})


def predict(request):
    probabilities = calculate_probabilities(request)

    if not probabilities:
        return render(request, "predictions.html", {"error": "No data available for predictions."})
    print("Monte Carlo Running")
    monte_carlo_results = monte_carlo_simulation(probabilities)
    print("Random Forest Prediction Running")
    random_forest_prediction = train_random_forest(request)
    print(random_forest_prediction)
    print("LSTM MODEL Running")
    lstm_prediction = train_lstm(request)
    print(lstm_prediction)
    print("Trend Analysis Running")
    trend_analysis_results = trend_analysis()

    combined_results = {
        "monte_carlo": monte_carlo_results,
        "random_forest": random_forest_prediction,
        "lstm": lstm_prediction,
        "trend_analysis": trend_analysis_results,
    }
    # print(combined_results)
    return render(request, "predictions.html", {"predictions": combined_results})


# Utility function to convert queryset to DataFrame
def get_card_data():
    data = list(Draw.objects.values('id', 'Club', 'Diamond', 'Heart', 'spade'))
    if not data:
        return None

    df = pd.DataFrame(data)

    # Convert multiple columns into a single "card" column
    df['card'] = df[['Club', 'Diamond', 'Heart', 'spade']].idxmax(
        axis=1)  # Get the highest value column as the drawn card

    return df[['id', 'card']]

# Efficient file handling
def handle_uploaded_file(file):
    file_path = default_storage.save(f"uploads/{file.name}", file)
    return file_path

#-------- IMPORT EXCEL DATA  ---------
@csrf_exempt
def import_excel_data(request):
    if request.method == 'POST' and request.FILES.get('file'):
        file_path = handle_uploaded_file(request.FILES['file'])
        try:
            df = pd.read_excel(file_path, usecols=['Draw Number', 'Card'])
            df.columns = ['draw_number', 'card']
            with transaction.atomic():
                Draw.objects.bulk_create([Draw(**row) for row in df.to_dict(orient='records')])
            return JsonResponse({'message': 'Data imported successfully'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)

#-------- PROBABILITIES CALCULATION  ---------
@csrf_exempt
def calculate_probabilities(request):
    df = get_card_data()
    if df is None:
        return JsonResponse({'error': 'No data available'}, status=400)

    probabilities = df['card'].value_counts(normalize=True).to_dict()
    return probabilities
    # return JsonResponse({'probabilities': probabilities})

#-------- TRAIN RANDOM FOREST  ---------
@csrf_exempt
def train_random_forest(request):
    print("Train Random Forest Running")

    df = get_card_data()
    if df is None:
        return JsonResponse({'error': 'No data available'}, status=400)

    # Encoding categorical card values
    df['card'] = df['card'].astype(str)  # Ensure all values are string type
    unique_cards = sorted(df['card'].unique())  # Get unique card values

    card_mapping = {card: idx for idx, card in enumerate(unique_cards)}
    df['card'] = df['card'].map(card_mapping)

    # Encode previous card
    df['prev_card'] = df['card'].shift(1).fillna(0).astype(int)
    X, y = df[['prev_card']], df['card']

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the trained model
    joblib.dump(model, 'random_forest_model.pkl')
    print('-'*50)
    print(card_mapping)
    print('-' * 50)
    return card_mapping
    # return JsonResponse({'message': 'Random Forest trained successfully', 'card_mapping': card_mapping})

#-------- RANDOM FOREST PREDECTION  ---------
@csrf_exempt
def predict_random_forest(request):
    if not os.path.exists('random_forest_model.pkl'):
        return JsonResponse({'error': 'Model not trained'}, status=400)

    model = joblib.load('random_forest_model.pkl')
    df = get_card_data()
    if df is None:
        return JsonResponse({'error': 'No data available'}, status=400)

    last_card = df.iloc[-1]['card']
    prediction = model.predict([[last_card]])[0]

    return JsonResponse({'prediction': prediction})

#-------- LSTM MODEL TRAINING  ---------
@csrf_exempt
def train_lstm(request):
    df = get_card_data()
    if df is None:
        return JsonResponse({'error': 'No data available'}, status=400)

    # Encode categorical values (Convert card names to numbers)
    df['card'] = df['card'].astype(str)  # Ensure all values are string
    unique_cards = sorted(df['card'].unique())  # Sort for consistent mapping
    card_mapping = {card: idx for idx, card in enumerate(unique_cards)}
    df['card'] = df['card'].map(card_mapping)  # Convert names to numbers

    # Define sequence length
    sequence_length = 5

    # Create sequences
    sequences = [df['card'].iloc[i - sequence_length:i].values for i in range(sequence_length, len(df))]

    # Convert sequences into a padded format
    X = pad_sequences(sequences, maxlen=sequence_length, padding='pre')
    y = df['card'][sequence_length:].values

    # Expand dimensions for LSTM input shape
    X = np.expand_dims(X, axis=-1)

    # Define LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
        Dense(50, activation='relu'),
        Dense(len(unique_cards), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X, y, epochs=20, batch_size=16, verbose=1)

    # Save the trained model
    model.save('lstm_model.h5')

    return JsonResponse({'message': 'LSTM trained successfully', 'card_mapping': card_mapping})

# -------- LSTM PREDICTION  ---------
@csrf_exempt
def predict_lstm(request):
    if not os.path.exists('lstm_model.h5'):
        return JsonResponse({'error': 'Model not trained'}, status=400)

    model = load_model('lstm_model.h5')
    df = get_card_data()
    if df is None:
        return JsonResponse({'error': 'No data available'}, status=400)

    sequence_length = 5
    last_sequence = df['card'].iloc[-sequence_length:].values.reshape(1, sequence_length, 1)
    prediction = np.argmax(model.predict(last_sequence))
    return JsonResponse({'prediction': int(prediction)})

#-------- MONTE CARLO SIMULATION  ---------
@csrf_exempt
def monte_carlo_simulation(probabilities, steps=7, num_simulations=1000):

    cards = list(probabilities.keys())
    weights = [probabilities[card] for card in cards]
    predictions = []

    for _ in range(num_simulations):
        future_draws = []
        for _ in range(steps):
            draw = random.choices(cards, weights=weights, k=2)
            future_draws.append(tuple(sorted(draw)))
        predictions.append(future_draws)

    aggregated_results = []
    for step in range(steps):
        step_counts = {}
        for sim in predictions:
            pair = sim[step]
            step_counts[pair] = step_counts.get(pair, 0) + 1
        step_results = [
            {"cards": pair, "probability": count / num_simulations}
            for pair, count in step_counts.items()
        ]
        step_results.sort(key=lambda x: x["probability"], reverse=True)
        aggregated_results.append(step_results[:3])
    return aggregated_results

#-------- TREND ANALYSIS  ---------
def trend_analysis():
    draws = Draw.objects.all()
    if len(draws) < 20:
        return None

    trend_counts = {}
    for draw in draws:
        cards = (draw.spade, draw.Heart, draw.Diamond, draw.Club)
        for card in cards:
            trend_counts[card] = trend_counts.get(card, 0) + 1

    sorted_trends = sorted(trend_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_trends[:5]
