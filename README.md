# Card Prediction Project

This is a Django-based project for card predictions using historical data and machine learning algorithms.

## 🚀 Features
- ✅ Add new draws to the database
- ✅ Calculate probabilities using historical data
- ✅ Perform predictions using:
  - Monte Carlo Simulations
  - Random Forest Classifier
  - LSTM Neural Networks
- ✅ Error Handling and Data Validation
- ✅ Model Overfitting Mitigation

## 📋 Requirements
- Python 3.9+
- Django 4.0+
- PostgreSQL
- TensorFlow
- scikit-learn
- NumPy & Pandas

## ⚙️ Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd card_prediction_project
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   - Create a `.env` file in the root directory.
   - Add database credentials and other environment settings.

4. **Run migrations:**
   ```bash
   python manage.py migrate
   ```

5. **Start the server:**
   ```bash
   python manage.py runserver
   ```

## 🧪 Testing
Run tests to ensure everything works:
```bash
python manage.py test
```

## 📊 Machine Learning Models
- **Random Forest:** For classification based on historical draws.
- **LSTM:** For sequence prediction of card patterns.
- **Monte Carlo Simulation:** To model randomness in card draws.

## 🗂️ Project Structure
```plaintext
card_prediction_project/
├── card_app/
│   ├── migrations/
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   └── templates/
├── ml_models/
│   ├── lstm_model.py
│   ├── random_forest_model.py
│   └── predict.py
├── dataset/
│   └── card_data.csv
├── requirements.txt
└── README.md
```

## 🛠️ Contributing
1. Fork the repo
2. Create your feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a pull request

## 📄 License
Distributed under the MIT License.
