import os

from django.shortcuts import render, redirect
from sklearn.utils import compute_class_weight

from .models import Draw
from .forms import DrawForm
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout ,Reshape,BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.core.exceptions import ValidationError
from tensorflow.keras.utils import to_categorical
from collections import Counter
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
    print("Monte Carlo Running")
    monte_carlo_results = monte_carlo_simulation(probabilities)
    print("Random Forest Prediction Running")
    random_forest_prediction = train_random_forest()
    print("LSTM MODEL Running")
    lstm_prediction = train_lstm()
    print("Trend Analysis Running")
    trend_analysis_results = trend_analysis()

    combined_results = {
        "monte_carlo": monte_carlo_results,
        "random_forest": random_forest_prediction,
        "lstm": lstm_prediction,
        "trend_analysis": trend_analysis_results,
    }
    print(combined_results)
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

    # Optimized data preparation using list comprehension
    data = np.array([
        [card_map[draw.spade], card_map[draw.Heart], card_map[draw.Diamond], card_map[draw.Club]]
        for draw in draws
    ])

    # Handling missing data
    data = data[~np.isnan(data).any(axis=1)]  # Remove rows with NaN values

    # Feature Scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Train-Test Split
    X, y = data_scaled[:-1], data_scaled[1:]

    # Check class distribution
    class_counts = Counter(y[:, 0])  # Use first column for class counting
    valid_classes = [cls for cls, count in class_counts.items() if count >= 2]

    # Filter out underrepresented classes
    mask = np.isin(y[:, 0], valid_classes)  # Apply mask based on first column
    X, y = X[mask], y[mask]

    # Shape validation
    if X.shape[0] != y.shape[0]:
        print(f"❗ Data shape mismatch detected: X shape = {X.shape}, y shape = {y.shape}")
        return ["Data shape mismatch after filtering"]

    # Conditional stratification
    if len(valid_classes) >= 2:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y[:, 0]  # Stratify using first column
        )
    else:
        print("⚠️ Warning: Insufficient samples for stratification. Proceeding without stratify.")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Compute Class Weights BEFORE One-hot Encoding
    if len(np.unique(y_train[:, 0])) > 1:
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train[:, 0]), y=y_train[:, 0])
        class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train[:, 0]), class_weights)}

        # Generate sample weights
        sample_weights = np.array([class_weight_dict[label] for label in y_train[:, 0]])

        # Expand sample weights to match the shape of y_train
        sample_weights = np.expand_dims(sample_weights, axis=-1)
        sample_weights = np.tile(sample_weights, (1, y_train.shape[1]))
    else:
        class_weight_dict = None
        sample_weights = None  # Skip weighting if only one class exists

    # One-hot Encoding
    y_train = to_categorical(y_train, num_classes=len(cards))
    y_val = to_categorical(y_val, num_classes=len(cards))

    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    # Model Definition
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], 1)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(4 * len(cards), activation='softmax'),
        Reshape((4, len(cards)))
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

    # Apply sample weights if available
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50, batch_size=32,
        sample_weight=sample_weights if sample_weights is not None else None,
        verbose=1, callbacks=[lr_reduction]
    )

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.show()

    # Making Predictions
    if len(X_val) > 0:
        next_draw = model.predict(X_val[-1].reshape(1, X_val.shape[1], 1))
        predicted_cards = np.argmax(next_draw, axis=2)
        return predicted_cards.flatten().tolist()

    return ["No prediction (insufficient data)"]
#
# def train_lstm():
#     draws = Draw.objects.all()
#     if len(draws) < 10:
#         return None
#
#     cards = ["7", "8", "9", "10", "J", "Q", "K", "A"]
#     card_map = {card: i for i, card in enumerate(cards)}
#
#     # Optimized data preparation using list comprehension
#     data = np.array([
#         [card_map[draw.spade], card_map[draw.Heart], card_map[draw.Diamond], card_map[draw.Club]]
#         for draw in draws
#     ])
#
#     X, y = data[:-1], data[1:]
#     X = X.reshape((X.shape[0], X.shape[1], 1))
#     y = to_categorical(y, num_classes=len(cards))
#
#     # Optimized model with Bidirectional LSTM and BatchNormalization
#     model = Sequential([
#         Bidirectional(LSTM(128, return_sequences=True), input_shape=(X.shape[1], 1)),
#         BatchNormalization(),
#         Dropout(0.3),
#         LSTM(64, return_sequences=False),
#         BatchNormalization(),
#         Dropout(0.3),
#         Dense(64, activation='relu'),
#         Dense(4 * len(cards), activation='softmax'),
#         Reshape((4, len(cards)))
#     ])
#
#     model.compile(optimizer=Adam(learning_rate=0.001),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     # Adaptive learning rate reduction
#     lr_reduction = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-5)
#
#     history = model.fit(X, y, epochs=50, batch_size=32, verbose=1, callbacks=[lr_reduction])
#
#     # Visualization
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Loss')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='Accuracy')
#     plt.legend()
#     plt.show()
#
#     # Making predictions
#     if len(X) > 0:
#         next_draw = model.predict(X[-1].reshape(1, X.shape[1], 1))
#         predicted_cards = np.argmax(next_draw, axis=2)
#         return predicted_cards.flatten().tolist()
#
#     return ["No prediction (insufficient data)"]
#
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
#
#     # Reshape inputs
#     X = X.reshape((X.shape[0], X.shape[1], 1))
#
#     # One-hot encode the output labels
#     y = to_categorical(y, num_classes=len(cards))  # Shape: (batch_size, 4, 8)
#
#     model = Sequential([
#         LSTM(256, return_sequences=True, input_shape=(X.shape[1], 1)),
#         Dropout(0.3),
#         LSTM(128, return_sequences=False),
#         Dropout(0.3),
#         Dense(128, activation='relu'),
#         Dense(4 * len(cards), activation='softmax'),   # 4 × 8 = 32 outputs
#         Reshape((4, len(cards)))                       # Reshape to (4, 8)
#     ])
#
#     model.compile(optimizer=Adam(learning_rate=0.001),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     def scheduler(epoch, lr):
#         return lr * 0.95
#
#     callbacks = [LearningRateScheduler(scheduler)]
#
#     history = model.fit(X, y, epochs=100, batch_size=16, verbose=1, callbacks=callbacks)
#
#     plt.plot(history.history['loss'], label='Loss')
#     plt.legend()
#     plt.show()
#
#     plt.plot(history.history['accuracy'], label='Accuracy')
#     plt.legend()
#     plt.show()
#     print(len(X))
#     if len(X) > 0:
#         next_draw = model.predict(X[-1].reshape(1, X.shape[1], 1))
#         predicted_cards = np.argmax(next_draw, axis=2)  # Predict for each suit
#         return predicted_cards.flatten().tolist()
#
#     return ["No prediction (insufficient data)"]

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
        cards = (draw.spade, draw.Heart, draw.Diamond, draw.Club)
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
