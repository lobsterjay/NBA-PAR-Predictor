import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def fetch_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    points = []
    assists = []
    rebounds = []
    three_point_made = []

    for row in soup.select('#pgl_basic tbody tr'):
        if row.get('class') and 'thead' in row['class']:
            continue

        pts = row.select_one('td[data-stat="pts"]')
        ast = row.select_one('td[data-stat="ast"]')
        trb = row.select_one('td[data-stat="trb"]')
        fg3 = row.select_one('td[data-stat="fg3"]')

        if pts and ast and trb and fg3:
            points.append(int(pts.text))
            assists.append(int(ast.text))
            rebounds.append(int(trb.text))
            three_point_made.append(int(fg3.text))

    return {
        'points': points,
        'assists': assists,
        'rebounds': rebounds,
        '3P Made': three_point_made
    }

def prepare_data(sequence):
    return np.arange(len(sequence)).reshape(-1, 1), np.array(sequence).reshape(-1, 1)

from sklearn.model_selection import TimeSeriesSplit

def train_and_predict_next_number(X, y):
    model = make_pipeline(PolynomialFeatures(), LinearRegression())
    
    param_grid = {'polynomialfeatures__degree': np.arange(1, 6)}
    
    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=3) if len(y) >= 3 else None
    
    grid_search = GridSearchCV(model, param_grid, cv=tscv)
    grid_search.fit(X, y)
    
    next_index = len(y)
    next_number_prediction = grid_search.best_estimator_.predict([[next_index]])
    return max(0, next_number_prediction[0][0])


url = 'https://www.basketball-reference.com/players/e/edwaran01/gamelog/2023'
fetched_data = fetch_data(url)

sequence_sets = {
    'Anthony Edwards': [
        {'data': fetched_data['points'], 'label': 'points'},
        {'data': fetched_data['assists'], 'label': 'assists'},
        {'data': fetched_data['rebounds'], 'label': 'rebounds'},
        {'data': fetched_data['3P Made'], 'label': '3P Made'}
    ],
    # Add more sets of input sequences with custom unique names as needed
}

print("The predicted next numbers in the sequence sets are:")

for set_key, sequences in sequence_sets.items():
    print(f"\n{set_key}:")
    for seq in sequences:
        X, y = prepare_data(seq['data'])
        next_number_prediction = train_and_predict_next_number(X, y)
        print(f"  {seq['label'].capitalize()}: {next_number_prediction:.2f}")
