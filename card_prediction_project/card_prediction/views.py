import json
import os
import random

import numpy as np
import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.core.exceptions import ValidationError
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import (Dense, LSTM)
from tensorflow.keras.models import Sequential

from .forms import DrawForm
from .models import Draw
from .models import Prediction


def predict_view(request):
    return render(request, "predictions.html")


# --- Add a draw to the database ---
def add_draw(request):
    print("ADD DRAW")
    if request.method == "POST":
        form = DrawForm(request.POST)
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
    print("PREDICT MODEL")
    probabilities = calculate_probabilities(request)

    if not probabilities:
        return render(request, "predictions.html", {"error": "No data available for predictions."})
    print("Monte Carlo Running")
    monte_carlo_results = monte_carlo_simulation(probabilities)
    print("Random Forest Prediction Running")
    random_forest_prediction_json = train_random_forest(request)
    random_forest_prediction = json.loads(random_forest_prediction_json.content)
    print("LSTM MODEL Running")
    lstm_prediction_json = train_and_predict_lstm(request)
    lstm_prediction = json.loads(lstm_prediction_json.content)
    print(lstm_prediction)
    # print("Trend Analysis Running")
    # trend_analysis_results = trend_analysis()

    combined_results = {
        "monte_carlo": monte_carlo_results,
        "random_forest": random_forest_prediction['predicted_cards'],
        "lstm": lstm_prediction['predicted_cards'],
        # "trend_analysis": trend_analysis_results,
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
# #-------- FUNCTION TO IMPORT EXCEL FILE INTO DATABASE ---------
def import_excel_data(request):
    # Path to the pre-uploaded Excel file
    file_path = os.path.join(settings.BASE_DIR, os.path.dirname(__file__), "Database_data.xlsx")

    if os.path.exists(file_path):
        try:
            # Read the Excel file
            df = pd.read_excel(file_path, engine="openpyxl", header=0)
            # Validate that the expected columns exist

            if not all(col in df.columns for col in ['Club', 'Diamond', 'Heart', 'spade']):
                raise ValidationError("Excel file must contain 'spade', 'heart', 'diamond', and 'club' columns.")

            # Iterate over the rows of the DataFrame and create Draw objects
            print("Please wait inserting Data")
            # Prepare a list of Draw objects to bulk create
            draw_objects = []
            for index, row in df.iterrows():
                try:
                    Draw.objects.create(
                            Club = row['Club'],
                            Diamond = row['Diamond'],
                            Heart=row['Heart'],
                            spade=row['spade'],
                        )
                except Exception as e:
                    print(f"Error inserting row {index}: {e}")
                    exit()

            # Bulk insert the data into the database
            Draw.objects.bulk_create(draw_objects)

            messages.success(request, "Data imported successfully!")

        except ValidationError as ve:
            messages.error(request, f"Validation Error: {ve}")
        except Exception as e:
            messages.error(request, f"Error-->: {e}")

    else:
        messages.error(request, "The specified Excel file does not exist.")

    return render(request, "import_excel.html")  # Render a page to show the result

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
    print("Training Random Forest Model...")

    # Define full 52-card deck
    suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
    ranks = list(range(1, 14))  # Ace (1), 2-10, Jack (11), Queen (12), King (13)

    # Generate all 52 cards
    full_deck = [{"card_rank": rank, "card_suit": suit} for suit in suits for rank in ranks]

    # Convert to DataFrame
    df = pd.DataFrame(full_deck)

    # Encode suits as numbers
    suit_encoder = LabelEncoder()
    df['card_suit_encoded'] = suit_encoder.fit_transform(df['card_suit'])

    # Add color feature (Red = 1, Black = 0)
    df['card_color'] = df['card_suit'].apply(lambda x: 1 if x in ["Hearts", "Diamonds"] else 0)

    # Features (X) and Target (y)
    X = df[['card_rank', 'card_color']]
    y = df['card_suit_encoded']

    # Split data into train and test (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the RandomForest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict card suits
    y_pred = model.predict(X_test)
    predicted_suits = suit_encoder.inverse_transform(y_pred)  # Convert numbers back to suit names

    # 🔄 Convert numeric ranks back to readable names (Ace, Jack, Queen, King)
    rank_mapping = {1: "Ace", 11: "Jack", 12: "Queen", 13: "King"}
    predicted_cards = [
        f"{rank_mapping.get(X_test.iloc[i]['card_rank'], int(X_test.iloc[i]['card_rank']))} of {predicted_suits[i]}"
        for i in range(len(y_pred))
    ]

    # 🔥 Limit predictions to only 2-4 cards
    predicted_cards = predicted_cards[:random.randint(2, 4)]

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100  # Convert to percentage

    print(f"Model Accuracy: {accuracy:.2f}%")

    return JsonResponse({
        'message': 'Random Forest trained successfully',
        'accuracy': accuracy,
        'predicted_cards': predicted_cards  # Returns 2-4 cards in "10 of Hearts" format
    })


# -------- LSTM MODEL   ---------
@csrf_exempt
def train_and_predict_lstm(request):
    print("Fetching Dataset...")

    # 🔥 Fetch dataset
    df = get_card_data()
    if df is None or df.empty:
        return JsonResponse({'error': 'No data available'}, status=400)

    # 🔥 Print dataset preview for debugging
    print("Raw Dataset:\n", df.head())

    # 🔥 Normalize suit names (fix casing, trim spaces)
    df['card'] = df['card'].str.strip().str.lower().str.title()

    if df.empty:
        return JsonResponse({'error': 'Dataset does not contain valid suits'}, status=400)

    # 🔥 Assign random ranks (since dataset lacks them)
    df['card_rank'] = np.random.randint(1, 14, size=len(df))  # Ace (1) to King (13)

    # 🔥 Encode suits as numbers
    suit_encoder = LabelEncoder()
    df['card_suit_encoded'] = suit_encoder.fit_transform(df['card'])

    # 🔥 Encode colors (Red = 1, Black = 0)
    df['card_color'] = df['card'].apply(lambda x: 1 if x in ['Hearts', 'Diamonds'] else 0)

    # 🔥 Prepare sequences (Using last 3 cards to predict the next)
    sequence_length = 3
    sequences, targets = [], []

    for i in range(len(df) - sequence_length):
        seq = df[['card_rank', 'card_suit_encoded', 'card_color']].iloc[i:i + sequence_length].values
        target = df[['card_rank', 'card_suit_encoded', 'card_color']].iloc[i + sequence_length].values
        sequences.append(seq)
        targets.append(target)

    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(targets)

    # 🔥 Train-Test Split (80-20)
    if len(X) < 10:  # Ensure enough data for training
        return JsonResponse({'error': 'Not enough data to train the model'}, status=400)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🔥 Define the LSTM Model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(sequence_length, 3)),
        LSTM(64),
        Dense(3, activation='linear')  # Predicting Rank, Suit, and Color
    ])

    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # Train Model
    model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=1)

    # 🔥 Predict Multiple Cards
    num_predictions = np.random.randint(2, 5)  # Randomly predict between 2-4 cards
    predicted_cards = []

    for i in range(num_predictions):
        last_sequence = X_test[i]  # Use multiple test examples for prediction
        prediction = model.predict(np.array([last_sequence]))[0]

        # Convert predictions back to readable format
        rank_mapping = {1: "Ace", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
                        10: "10", 11: "Jack", 12: "Queen", 13: "King"}
        predicted_rank = int(round(prediction[0]))
        predicted_suit = suit_encoder.inverse_transform([int(round(prediction[1]))])[0]
        predicted_color = "Red" if round(prediction[2]) == 1 else "Black"

        predicted_cards.append(f"{rank_mapping.get(predicted_rank, predicted_rank)} of {predicted_suit} ")

    return JsonResponse({
        'message': 'LSTM trained and predicted successfully',
        'predicted_cards': predicted_cards  # 🔥 Now returns multiple predicted cards
    })

#-------- MONTE CARLO SIMULATION  ---------
@csrf_exempt
def monte_carlo_simulation(probabilities, steps=7, num_simulations=1000):
    print("MONTE CARLO SIMULATON")
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

#-------- SAVE PREDICTION DATA  ---------
@csrf_exempt
def save_predictions(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)  # ✅ Get JSON from Fetch API
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)
        # Save data to DB
        Prediction.objects.create(
            monte_carlo=data.get('monte_carlo'),
            random_forest=data.get('random_forest'),
            lstm=data.get('lstm'),
            trend_analysis=data.get('trend_analysis')
        )

        return JsonResponse({'message': 'Predictions saved successfully!'})

    return JsonResponse({'error': 'Invalid request method'}, status=400)
#
# @csrf_exempt
# def save_predictions(request):
#     if request.method == 'POST':
#         predictions_json = request.POST.get('predictions')
#         predictions = json.loads(predictions_json)
#
#         # Save to DB
#         Prediction.objects.create(
#             monte_carlo=predictions.get('monte_carlo'),
#             random_forest=predictions.get('random_forest', {}).get('prediction'),
#             lstm=predictions.get('lstm', {}).get('prediction'),
#             trend_analysis=predictions.get('trend_analysis')
#         )
#
#         return JsonResponse({'message': 'Predictions saved successfully!'})
#
#     return JsonResponse({'error': 'Invalid request'}, status=400)