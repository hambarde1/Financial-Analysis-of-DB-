

# Stock Price Prediction Using RNN, LSTM, Stacked LSTM, and Bidirectional LSTM

## Overview
This project aims to predict stock prices using various recurrent neural network models. Specifically, we utilize historical data from Deutsche Bank and implement RNN, LSTM, Stacked LSTM, and Bidirectional LSTM models. The performance of each model is evaluated and compared to determine which architecture is most effective for stock price prediction.

## Project Structure
- **Data Collection**: The data is fetched using the `yfinance` library for the past 5 years.
- **Model Implementation**: Four types of models are implemented:
  - Recurrent Neural Network (RNN)
  - Long Short-Term Memory (LSTM)
  - Stacked LSTM
  - Bidirectional LSTM
- **Evaluation Metrics**: Mean Absolute Error (MAE) and Mean Squared Error (MSE) are used to assess the models' performance.
- **Visualization**: The code includes plots comparing actual stock prices with the predictions from each model.

## Data and Preprocessing
The dataset includes adjusted closing prices of Deutsche Bank's stock over a 5-year period. Data preprocessing steps include:
- Normalization using `MinMaxScaler`.
- Creation of training and testing sets.
- Transformation into a format suitable for time-series model training.

## Model Details
### 1. Recurrent Neural Network (RNN)
- A simple RNN with one layer.
- Used as a baseline for comparison.

### 2. Long Short-Term Memory (LSTM)
- An LSTM layer designed to capture long-term dependencies in the data.

### 3. Stacked LSTM
- Multiple LSTM layers stacked on top of each other for deeper learning.

### 4. Bidirectional LSTM
- Processes the input sequence in both forward and backward directions, providing a more comprehensive understanding of the data context.

## Results
The table below compares the MAE (Mean Absolute Error) for each model:

| Model               | MAE   |
|---------------------|-------|
| RNN                 | 0.27  |
| LSTM                | 0.21  |
| Stacked LSTM        | 0.20  |
| Bidirectional LSTM  | 0.17  |

**Bidirectional LSTM** achieved the lowest MAE, demonstrating superior accuracy in stock price prediction compared to the other models.

## Visualizations
The code includes a plot that displays the actual stock prices alongside the predicted prices from all models to visually compare their performance.

## How to Run the Code
### Prerequisites
Ensure that Python and the following libraries are installed:
- `yfinance`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`

### Installation
Install the required libraries using pip:
```bash
pip install yfinance numpy pandas matplotlib scikit-learn tensorflow
```

### Running the Code
1. Clone the repository or copy the code to your local environment.
2. Run the Python script to train the models and generate the results.

## Key Takeaways
- **RNN**: Served as the baseline with moderate accuracy.
- **LSTM**: Showed improved performance by capturing longer dependencies.
- **Stacked LSTM**: Provided a slight improvement over the standard LSTM by learning deeper representations.
- **Bidirectional LSTM**: Outperformed all other models by leveraging both past and future context for each point in the sequence.

## Future Work
To further improve the predictive accuracy and robustness, consider:
- Hyperparameter tuning for all models.
- Integrating attention mechanisms with LSTM.
- Using more advanced architectures such as hybrid CNN-LSTM models or transformers for time-series forecasting.
