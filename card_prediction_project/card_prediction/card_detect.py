from django.shortcuts import render, redirect
# from .forms import DrawForm
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt


# from .models import Draw


def add_draw(request):
    if request.method == "POST":
        form = DrawForm(request.POST)
        if form.is_valid():
            Draw.objects.create(
                spade=form.cleaned_data["spade"],
                heart=form.cleaned_data["heart"],
                diamond=form.cleaned_data["diamond"],
                club=form.cleaned_data["club"],
            )
            return redirect("home")
    else:
        form = DrawForm()
    return render(request, "add_draw.html", {"form": form})


def calculate_probabilities():
    draws = Draw.objects.all()
    if not draws.exists():
        return {}

    cards = ["7", "8", "9", "10", "J", "Q", "K", "A"]
    suits = ["Spade", "Heart", "Diamond", "Club"]
    card_counts = {f"{card} ({suit})": 0 for card in cards for suit in suits}

    for draw in draws:
        card_counts[f"{draw.spade} (Spade)"] += 1
        card_counts[f"{draw.heart} (Heart)"] += 1
        card_counts[f"{draw.diamond} (Diamond)"] += 1
        card_counts[f"{draw.club} (Club)"] += 1

    total_cards = len(draws) * 4
    probabilities = {card: count / total_cards for card, count in card_counts.items()}
    return probabilities


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
            card_map[draw.heart],
            card_map[draw.diamond],
            card_map[draw.club]
        ])

    X = np.array(data[:-1])
    y = np.array(data[1:])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

    if len(X_test) > 0:
        next_draw = model.predict([X_test[0]])
        return next_draw.tolist()
    return ["No prediction (insufficient data)"]


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
            card_map[draw.heart],
            card_map[draw.diamond],
            card_map[draw.club]
        ])

    X = np.array(data[:-1])
    y = np.array(data[1:])
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(len(cards) * 4, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
        return next_draw.tolist()
    return ["No prediction (insufficient data)"]


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


def predict(request):
    probabilities = calculate_probabilities()
    if not probabilities:
        return render(request, "predictions.html", {"error": "No data available for predictions."})

    monte_carlo_results = monte_carlo_simulation(probabilities)
    random_forest_prediction = train_random_forest()
    lstm_prediction = train_lstm()
    trend_analysis_results = trend_analysis()

    combined_results = {
        "monte_carlo": monte_carlo_results,
        "random_forest": random_forest_prediction,
        "lstm": lstm_prediction,
        "trend_analysis": trend_analysis_results,
    }

    return render(request, "predictions.html", {"predictions": combined_results})


if __name__ == "__main__":
    add_draw()
