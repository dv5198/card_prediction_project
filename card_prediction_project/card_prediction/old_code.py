#
# #-------- FUNCTION TO IMPORT EXCEL FILE INTO DATABASE ---------
# def import_excel_data(request):
#     # Path to the pre-uploaded Excel file
#     file_path = os.path.join(settings.BASE_DIR, os.path.dirname(__file__), "Database_data.xlsx")
#
#     if os.path.exists(file_path):
#         try:
#             # Read the Excel file
#             df = pd.read_excel(file_path, engine="openpyxl", header=0)
#             # Validate that the expected columns exist
#
#             if not all(col in df.columns for col in ['Club', 'Diamond', 'Heart', 'spade']):
#                 raise ValidationError("Excel file must contain 'spade', 'heart', 'diamond', and 'club' columns.")
#
#             # Iterate over the rows of the DataFrame and create Draw objects
#             print("Please wait inserting Data")
#             # Prepare a list of Draw objects to bulk create
#             draw_objects = []
#             for index, row in df.iterrows():
#                 try:
#                     Draw.objects.create(
#                             Club = row['Club'],
#                             Diamond = row['Diamond'],
#                             Heart=row['Heart'],
#                             spade=row['spade'],
#                         )
#                 except Exception as e:
#                     print(f"Error inserting row {index}: {e}")
#                     exit()
#
#             # Bulk insert the data into the database
#             Draw.objects.bulk_create(draw_objects)
#
#             messages.success(request, "Data imported successfully!")
#
#         except ValidationError as ve:
#             messages.error(request, f"Validation Error: {ve}")
#         except Exception as e:
#             messages.error(request, f"Error-->: {e}")
#
#     else:
#         messages.error(request, "The specified Excel file does not exist.")
#
#     return render(request, "import_excel.html")  # Render a page to show the result
#
#-------- FUNCTION TO CALCULATE PROBABILITIES ---------
# def calculate_probabilities():
#     draws = Draw.objects.all()
#
#     if not draws.exists():
#         return {}
#
#     cards = ["7", "8", "9", "10", "J", "Q", "K", "A"]
#     suits = ["Spade", "Heart", "Diamond", "Club"]
#     card_counts = {f"{card} ({suit})": 0 for card in cards for suit in suits}
#
#     for draw in draws:
#         card_counts[f"{draw.spade} (Spade)"] += 1
#         card_counts[f"{draw.Heart} (Heart)"] += 1
#         card_counts[f"{draw.Diamond} (Diamond)"] += 1
#         card_counts[f"{draw.Club} (Club)"] += 1
#
#     total_cards = len(draws) * 4
#     probabilities = {card: count / total_cards for card, count in card_counts.items()}
#     return probabilities


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
#     # Handling missing data
#     data = data[~np.isnan(data).any(axis=1)]  # Remove rows with NaN values
#
#     # Feature Scaling
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(data)
#
#     # Train-Test Split
#     X, y = data_scaled[:-1], data_scaled[1:]
#
#     # Check class distribution
#     class_counts = Counter(y[:, 0])  # Use first column for class counting
#     valid_classes = [cls for cls, count in class_counts.items() if count >= 2]
#
#     # Filter out underrepresented classes
#     mask = np.isin(y[:, 0], valid_classes)  # Apply mask based on first column
#     X, y = X[mask], y[mask]
#
#     # Shape validation
#     if X.shape[0] != y.shape[0]:
#         print(f"❗ Data shape mismatch detected: X shape = {X.shape}, y shape = {y.shape}")
#         return ["Data shape mismatch after filtering"]
#
#     # Conditional stratification
#     if len(valid_classes) >= 2:
#         X_train, X_val, y_train, y_val = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=y[:, 0]  # Stratify using first column
#         )
#     else:
#         print("⚠️ Warning: Insufficient samples for stratification. Proceeding without stratify.")
#         X_train, X_val, y_train, y_val = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )
#
#     # Compute Class Weights BEFORE One-hot Encoding
#     if len(np.unique(y_train[:, 0])) > 1:
#         class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train[:, 0]), y=y_train[:, 0])
#         class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train[:, 0]), class_weights)}
#
#         # Generate sample weights
#         sample_weights = np.array([class_weight_dict[label] for label in y_train[:, 0]])
#
#         # Expand sample weights to match the shape of y_train
#         sample_weights = np.expand_dims(sample_weights, axis=-1)
#         sample_weights = np.tile(sample_weights, (1, y_train.shape[1]))
#     else:
#         class_weight_dict = None
#         sample_weights = None  # Skip weighting if only one class exists
#
#     # One-hot Encoding
#     y_train = to_categorical(y_train, num_classes=len(cards))
#     y_val = to_categorical(y_val, num_classes=len(cards))
#
#     # Reshape for LSTM
#     X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
#     X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
#
#     # Model Definition
#     model = Sequential([
#         Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], 1)),
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
#     lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
#
#     # Apply sample weights if available
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=50, batch_size=32,
#         sample_weight=sample_weights if sample_weights is not None else None,
#         verbose=1, callbacks=[lr_reduction]
#     )
#
#     # Visualization
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Loss')
#     plt.plot(history.history['val_loss'], label='Val Loss')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Val Accuracy')
#     plt.legend()
#     plt.show()
#
#     # Making Predictions
#     if len(X_val) > 0:
#         next_draw = model.predict(X_val[-1].reshape(1, X_val.shape[1], 1))
#         predicted_cards = np.argmax(next_draw, axis=2)
#         return predicted_cards.flatten().tolist()
#
#     return ["No prediction (insufficient data)"]
#

#
# def train_random_forest():
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
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # model = RandomForestClassifier(n_estimators=200, random_state=42)
#     # model.fit(X_train, y_train)
#
#     # Wrap RandomForestClassifier with MultiOutputClassifier
#     model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
#     model.fit(X_train, y_train)
#
#     accuracy = model.score(X_test, y_test)
#     print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
#
#     if len(X_test) > 0:
#         next_draw = model.predict([X_test[0]])
#         return next_draw.tolist()
#     return ["No prediction (insufficient data)"]
