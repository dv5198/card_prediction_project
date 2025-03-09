# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Embedding
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
#
# # 🔥 Define a full 52-card deck
# suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
# ranks = {1: "Ace", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
#          10: "10", 11: "Jack", 12: "Queen", 13: "King"}
#
# # Generate all 52 cards
# full_deck = []
# for suit in suits:
#     for rank_num, rank_name in ranks.items():
#         color = "Red" if suit in ["Hearts", "Diamonds"] else "Black"
#         full_deck.append({"card_rank": rank_num, "card_suit": suit, "card_color": color})
#
# # Convert to DataFrame
# df = pd.DataFrame(full_deck)
#
# # 🔄 Encode suits & ranks into numbers
# suit_encoder = LabelEncoder()
# df["card_suit_encoded"] = suit_encoder.fit_transform(df["card_suit"])
#
# # 🔄 Encode colors as binary (Red = 1, Black = 0)
# df["card_color_encoded"] = df["card_color"].map({"Red": 1, "Black": 0})
#
# # 🔄 Prepare sequences (Using last 3 cards to predict the next)
# sequence_length = 3
# sequences = []
# targets = []
#
# for i in range(len(df) - sequence_length):
#     seq = df[["card_rank", "card_suit_encoded", "card_color_encoded"]].iloc[i:i+sequence_length].values
#     target = df[["card_rank", "card_suit_encoded", "card_color_encoded"]].iloc[i+sequence_length].values
#     sequences.append(seq)
#     targets.append(target)
#
# # Convert to numpy arrays
# X = np.array(sequences)
# y = np.array(targets)
#
# # 🔄 Train-Test Split (80-20)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 🔄 Define the LSTM Model
# model = Sequential([
#     LSTM(50, return_sequences=True, input_shape=(sequence_length, 3)),
#     LSTM(50),
#     Dense(3, activation='linear')  # Predicting Rank, Suit, and Color
# ])
#
# # Compile model
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#
# # Train Model
# model.fit(X_train, y_train, epochs=50, batch_size=2, validation_data=(X_test, y_test), verbose=1)
#
# # 🔄 Predict Next Card
# def predict_next_card(input_sequence):
#     input_sequence = np.array([input_sequence])
#     prediction = model.predict(input_sequence)[0]
#
#     # Convert predicted values back to labels
#     predicted_rank = int(round(prediction[0]))
#     predicted_suit = suit_encoder.inverse_transform([int(round(prediction[1]))])[0]
#     predicted_color = "Red" if round(prediction[2]) == 1 else "Black"
#
#     return f"{ranks.get(predicted_rank, predicted_rank)} of {predicted_suit} ({predicted_color})"
#
# # 🔥 Example Prediction (Using last 3 cards)
# sample_sequence = X_test[0]
# predicted_card = predict_next_card(sample_sequence)
# print(f"Predicted next card: {predicted_card}")
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential


@csrf_exempt
def train_and_predict_lstm(request):
    print("Fetching Dataset...")

    # 🔥 Fetch dataset
    df = get_card_data()
    if df is None or df.empty:
        return JsonResponse({'error': 'No data available'}, status=400)

    # 🔥 Define possible suits and ranks
    card_suits = ['Clubs', 'Spades', 'Hearts', 'Diamonds']
    card_ranks = {1: "Ace", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
                  10: "10", 11: "Jack", 12: "Queen", 13: "King"}

    # 🔥 Ensure dataset contains valid suits
    if not all(df['card'].isin(card_suits)):
        return JsonResponse({'error': 'Dataset must contain valid suits'}, status=400)

    # 🔥 Assign random ranks (since ranks are missing in dataset)
    df['card_rank'] = np.random.randint(1, 14, size=len(df))  # Random rank from 1 (Ace) to 13 (King)

    # 🔥 Encode suits as numbers
    suit_encoder = LabelEncoder()
    df['card_suit_encoded'] = suit_encoder.fit_transform(df['card'])

    # 🔥 Encode colors (Red = 1, Black = 0)
    df['card_color'] = df['card'].apply(lambda x: 1 if x in ['Hearts', 'Diamonds'] else 0)

    # 🔥 Prepare sequences (Using last 3 cards to predict the next)
    sequence_length = 3
    sequences = []
    targets = []

    for i in range(len(df) - sequence_length):
        seq = df[['card_rank', 'card_suit_encoded', 'card_color']].iloc[i:i + sequence_length].values
        target = df[['card_rank', 'card_suit_encoded', 'card_color']].iloc[i + sequence_length].values
        sequences.append(seq)
        targets.append(target)

    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(targets)

    # 🔥 Train-Test Split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🔥 Define the LSTM Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 3)),
        LSTM(50),
        Dense(3, activation='linear')  # Predicting Rank, Suit, and Color
    ])

    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # Train Model
    model.fit(X_train, y_train, epochs=50, batch_size=2, validation_data=(X_test, y_test), verbose=1)

    # 🔥 Predict Next Card
    last_sequence = X_test[0]  # Use the first test example for prediction
    prediction = model.predict(np.array([last_sequence]))[0]

    # Convert predictions back to readable format
    predicted_rank = int(round(prediction[0]))
    predicted_suit = suit_encoder.inverse_transform([int(round(prediction[1]))])[0]
    predicted_color = "Red" if round(prediction[2]) == 1 else "Black"

    predicted_card = f"{card_ranks.get(predicted_rank, predicted_rank)} of {predicted_suit} ({predicted_color})"

    return JsonResponse({
        'message': 'LSTM trained and predicted successfully',
        'predicted_card': predicted_card
    })
