# Card Prediction Project

This is a Django-based project for card predictions using historical data and machine learning algorithms.

## Features
- Add new draws to the database.
- Calculate probabilities using historical data.
- Perform predictions using Monte Carlo simulations, Random Forest, and LSTM.

## Requirements
- Python 3.9+
- Django 4.0+
- PostgreSQL

## Installation
1. Clone the repository:
   ```
   git clone <repo-url>
   cd card_prediction_project
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up the environment variables in `.env`.
4. Run migrations:
   ```
   python manage.py migrate
   ```
5. Start the server:
   ```
   python manage.py runserver