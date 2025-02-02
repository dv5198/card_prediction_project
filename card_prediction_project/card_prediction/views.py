import os

from django.shortcuts import render, redirect
from .models import Draw
from .forms import DrawForm
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout ,Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.core.exceptions import ValidationError
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
    probabilities = calculate_probabilities()

    if not probabilities:
        return render(request, "predictions.html", {"error": "No data available for predictions."})

    monte_carlo_results = monte_carlo_simulation(probabilities)
    # print(monte_carlo_results)
    # print("========================")
    random_forest_prediction = train_random_forest()
    print("=========   ",random_forest_prediction)
    lstm_prediction = train_lstm()
    trend_analysis_results = trend_analysis()

    combined_results = {
        "monte_carlo": monte_carlo_results,
        "random_forest": random_forest_prediction,
        "lstm": lstm_prediction,
        "trend_analysis": trend_analysis_results,
    }

    return render(request, "predictions.html", {"predictions": combined_results})

#-------- FUNCTION TO IMPORT EXCEL FILE INTO DATABASE ---------
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

#-------- FUNCTION TO CALCULATE PROBABILITIES ---------
def calculate_probabilities():
    draws = Draw.objects.all()

    if not draws.exists():
        return {}

    cards = ["7", "8", "9", "10", "J", "Q", "K", "A"]
    suits = ["Spade", "Heart", "Diamond", "Club"]
    card_counts = {f"{card} ({suit})": 0 for card in cards for suit in suits}

    for draw in draws:
        card_counts[f"{draw.spade} (Spade)"] += 1
        card_counts[f"{draw.Heart} (Heart)"] += 1
        card_counts[f"{draw.Diamond} (Diamond)"] += 1
        card_counts[f"{draw.Club} (Club)"] += 1

    total_cards = len(draws) * 4
    probabilities = {card: count / total_cards for card, count in card_counts.items()}
    return probabilities

#-------- MONTE CARLO SIMULATION  ---------
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

def train_lstm():
    draws = Draw.objects.all()
    if len(draws) < 10:
        return None

    cards = ["7", "8", "9", "10", "J", "Q", "K", "A"]
    card_map = {card: i for i, card in enumerate(cards)}

    data = []
    for draw in draws:
        data.append([
            card_map[draw.spade],
            card_map[draw.Heart],
            card_map[draw.Diamond],
            card_map[draw.Club]
        ])

    X = np.array(data[:-1])
    y = np.array(data[1:])

    # Reshape inputs
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # One-hot encode the output labels
    y = to_categorical(y, num_classes=len(cards))  # Shape: (batch_size, 4, 8)

    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(4 * len(cards), activation='softmax'),   # 4 × 8 = 32 outputs
        Reshape((4, len(cards)))                       # Reshape to (4, 8)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    def scheduler(epoch, lr):
        return lr * 0.95

    callbacks = [LearningRateScheduler(scheduler)]

    history = model.fit(X, y, epochs=100, batch_size=16, verbose=1, callbacks=callbacks)

    plt.plot(history.history['loss'], label='Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.legend()
    plt.show()

    if len(X) > 0:
        next_draw = model.predict(X[-1].reshape(1, X.shape[1], 1))
        predicted_cards = np.argmax(next_draw, axis=2)  # Predict for each suit
        return predicted_cards.flatten().tolist()

    return ["No prediction (insufficient data)"]

# def train_lstm():
#     draws = Draw.objects.all()
#     if len(draws) < 10:
#         return None
#
#     cards = ["7", "8", "9", "10", "J", "Q", "K", "A"]
#     card_map = {card: i for i, card in enumerate(cards)}
#
#     data = []
#     for draw in draws:
#         data.append([
#             card_map[draw.spade],
#             card_map[draw.Heart],
#             card_map[draw.Diamond],
#             card_map[draw.Club]
#         ])
#
#     X = np.array(data[:-1])
#     y = np.array(data[1:])
#     y = to_categorical(y, num_classes=len(cards))
#     X = X.reshape((X.shape[0], X.shape[1], 1))
#
#     model = Sequential([
#         LSTM(256, return_sequences=True, input_shape=(X.shape[1], 1)),
#         Dropout(0.3),
#         LSTM(128, return_sequences=False),
#         Dropout(0.3),
#         Dense(128, activation='relu'),
#         Dense(len(cards) * 4, activation='softmax')
#     ])
#
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
#     def scheduler(epoch, lr):
#         return lr * 0.95
#
#     callbacks = [LearningRateScheduler(scheduler)]
#     history = model.fit(X, y, epochs=100, batch_size=16, verbose=1, callbacks=callbacks)
#
#     plt.plot(history.history['loss'], label='Loss')
#     plt.legend()
#     plt.show()
#
#     plt.plot(history.history['accuracy'], label='Accuracy')
#     plt.legend()
#     plt.show()
#
#     if len(X) > 0:
#         next_draw = model.predict(X[-1].reshape(1, X.shape[1], 1))
#         return next_draw.tolist()
#     return ["No prediction (insufficient data)"]


def trend_analysis():
    draws = Draw.objects.all()
    if len(draws) < 20:
        return None

    trend_counts = {}
    for draw in draws:
        cards = (draw.spade, draw.heart, draw.diamond, draw.club)
        for card in cards:
            trend_counts[card] = trend_counts.get(card, 0) + 1

    sorted_trends = sorted(trend_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_trends[:5]


def train_random_forest():
    draws = Draw.objects.all()
    if len(draws) < 10:
        return None

    cards = ["7", "8", "9", "10", "J", "Q", "K", "A"]
    card_map = {card: i for i, card in enumerate(cards)}

    data = []
    for draw in draws:
        data.append([
            card_map[draw.spade],
            card_map[draw.Heart],
            card_map[draw.Diamond],
            card_map[draw.Club]
        ])

    X = np.array(data[:-1])
    y = np.array(data[1:])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model = RandomForestClassifier(n_estimators=200, random_state=42)
    # model.fit(X_train, y_train)

    # Wrap RandomForestClassifier with MultiOutputClassifier
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

    if len(X_test) > 0:
        next_draw = model.predict([X_test[0]])
        return next_draw.tolist()
    return ["No prediction (insufficient data)"]
