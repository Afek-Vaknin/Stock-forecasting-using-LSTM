# Tensorflow / Keras
import keras.models
from keras.models import Sequential     # for creating a linear stack of layers for our Neural Network
from keras import Input     # for instantiating a keras tensor
from keras.layers import Dense, LSTM  # for creating regular densely-connected NN layers and LSTM layers

# Data manipulation
import pandas as pd     # for data manipulation
import numpy as np  # for data manipulation
from math import floor
# Sklearn
from sklearn.model_selection import train_test_split    # for splitting the data into train and test samples
from sklearn.metrics import mean_squared_error  # for model evaluation metrics
from sklearn.preprocessing import MinMaxScaler  # for feature scaling

# Visualization
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Time libraries
import time
import datetime

# Database
import sqlite3


class LSTMPrediction:
    def __init__(self, symbol):
        """
        # step 1 - Receiving data
        # Step 2 - Scaling data
        # Step 3 - Use model to make predictions
        :param symbol: company's name
        """
        self.model = keras.models.load_model("LSTM")
        self.time_step = 36

        self.receive(symbol)
        self.scaling()
        self.predict_next_days()
        self.price_value()
        self.plot_price_prediction(symbol)

    def receive(self, company_symbol):
        first_day = datetime.date.today() - datetime.timedelta(days=200)
        self.yesterday = datetime.date.today() - datetime.timedelta(days=1)

        symbol = company_symbol
        period1 = int(time.mktime(first_day.timetuple()))
        period2 = int(time.mktime(self.yesterday.timetuple()))
        interval = '1d'

        query_string = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}" \
                       f"?period1={period1}&period2={period2}&interval={interval}&events=history" \
                       f"&includeAdjustedClose=true"

        self.df = pd.read_csv(query_string)
        self.df['MedPrice'] = self.df[['High', 'Low']].median(axis=1)

    def scaling(self):
        self.X = self.df[['MedPrice']]
        self.scaler = MinMaxScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

    def predict_next_days(self):
        self.X = self.scaler.transform(self.X)

        time_step_array = self.X[0:0 + self.time_step]
        for i in range(1, len(self.X) - self.time_step):
            time_step_array = np.append(time_step_array, self.X[i:i + self.time_step])

        time_step_array = np.reshape(time_step_array, (floor(len(time_step_array) / self.time_step), self.time_step, 1))

        self.df['MedPrice_prediction'] = np.append(np.zeros(self.time_step),
                                                   self.scaler.inverse_transform(self.model.predict(time_step_array)))

        # Get the last sequence in the data to start predictions
        inputs = time_step_array[-1:]
        prediction_list = []

        days = 5
        for i in range(days + 1):
            # Generate prediction and add it to the list
            prediction_list.append(list(self.model.predict(inputs)[0]))
            # Drop oldest and append the latest prediction
            inputs = np.append(inputs[:, 1:, :], [[prediction_list[i]]], axis=1)

        prediction_list = prediction_list[1:]

        self.newdf = pd.DataFrame(pd.date_range(start=self.yesterday, periods=days, freq='D'), columns=['Date'])
        self.newdf['MedPrice_prediction'] = self.scaler.inverse_transform(prediction_list)

        self.df2 = pd.concat([self.df, self.newdf])
        self.df2 = self.df2[41:]

    def price_value(self):
        """
        takes yesterday's price and today's predicted price and compare the two.
        send the right value (Higher/Lower/Same) to the client
        """

        yesterday = float(self.df.tail(1)['MedPrice_prediction'])  # predicted value
        today = float(self.newdf.head(1)['MedPrice_prediction'])  # predicted value

        if yesterday < today:
            self.value = "Higher"
        elif yesterday > today:
            self.value = "Lower"
        else:
            self.value = "Same"

    def plot_price_prediction(self, symbol):
        symbol = symbol.upper
        plt.subplots(figsize=(12, 8))  # set the figure size
        plt.plot(self.df2['Date'].astype(str), self.df2['MedPrice'], label='MedPrice')
        plt.plot(self.df2['Date'].astype(str), self.df2['MedPrice_prediction'], label='Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{symbol} - Predicted vs Actual Price')
        plt.legend()
        plt.xticks(rotation=60)
        ax = plt.gca()
        tick_spacing = 3  # show every other tick label
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        plt.show()

    def show_original_and_predict_graph(self, symbol):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df2['Date'][-365:],
                                 y=self.df2['MedPrice'][-365:],
                                 mode='lines',
                                 name='Median Price - Actual',
                                 opacity=0.8,
                                 line=dict(color='black', width=1)
                                 ))
        fig.add_trace(go.Scatter(x=self.df2['Date'][-365:],
                                 y=self.df2['MedPrice_prediction'][-365:],
                                 mode='lines',
                                 name='Median Price - Predicted',
                                 opacity=0.8,
                                 line=dict(color='red', width=1)
                                 ))

        # Change chart background color
        fig.update_layout(dict(plot_bgcolor='white'))

        # Update axes lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                         zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                         showline=True, linewidth=1, linecolor='black',
                         title='Date'
                         )

        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                         zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                         showline=True, linewidth=1, linecolor='black',
                         title='Price($)'
                         )

        # Set figure title
        fig.update_layout(title=dict(text=f"Median Daily Price of {symbol.upper()}",
                                     font=dict(color='black')),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                          )

        fig.show()


class TrainLSTM:
    def __init__(self, symbol):
        """
        # step 0 - Receiving data
        # Step 1 - Scaling data
        # Step 2 - Split dataset to training and testing samples
        # Step 3 - Prepare input X and target y arrays
        # Step 4 - Create the model's structure
        # Step 5 - Compile keras model
        # Step 6 - Fit keras model on the dataset
        :param symbol: company's name
        """
        self.time_step = 36
        self.receive(symbol)
        self.scaling()
        self.split()
        self.prep()
        self.architecture()
        self.compile()
        self.fit()
        # self.plot_loss(symbol)
        # self.performance_summary()
        self.model.save("LSTM")

    def receive(self, company_symbol):
        self.yesterday = datetime.date.today() - datetime.timedelta(days=1)

        symbol = company_symbol
        period1 = int(time.mktime(datetime.datetime(2017, 12, 31, 23, 59).timetuple()))
        period2 = int(time.mktime(self.yesterday.timetuple()))
        interval = '1d'

        query_string = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}" \
                       f"?period1={period1}&period2={period2}&interval={interval}&events=history" \
                       f"&includeAdjustedClose=true"

        self.df = pd.read_csv(query_string)
        self.df['MedPrice'] = self.df[['High', 'Low']].median(axis=1)

    def scaling(self):
        self.X = self.df[['MedPrice']]
        self.scaler = MinMaxScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

    def split(self):
        self.X_train, self.X_test = train_test_split(self.X_scaled, test_size=0.15, shuffle=False)

    def prep(self):
        self.X_train, self.y_train = self.prepare_data(self.X_train, self.time_step)
        self.X_test, self.y_test = self.prepare_data(self.X_test, self.time_step)

    def prepare_data(self, dataset, _time_step):
        """
        splits the data to an array which contains clusters of _time_step size,
        and create another array which contains the indices of the value we want to predict

        we take a _time_step index and predict the next - which is also the first of the next array
        example: _time_step = 7
                 [1,2,3,4,5,6,7] --(predict)--> 8
                 [8,9,10,11,12,13,14] --(predict)--> 15

        :param dataset: data to manipulate
        :param _time_step: length of a single sequence
        :return: x_tmp --> list of all training sets
                 y_tmp --> list of all tests sets
        """
        # Takes the indices which divided by _time_step's value
        sets = [k for k in range(_time_step, len(dataset), _time_step)]

        y_tmp = dataset[sets]
        sub_sequences = len(y_tmp)

        x_tmp = dataset[range(_time_step * sub_sequences)]
        x_tmp = np.reshape(x_tmp, (sub_sequences, _time_step, 1))
        return x_tmp, y_tmp

    def architecture(self):
        try:
            self.model = keras.models.load_model("LSTM")
        except:
            self.model = Sequential(name="LSTM-model")
            self.model.add(Input(shape=(self.time_step, 1), name='F-Input-Layer'))
            self.model.add(Dense(units=64, activation='relu', name='First-Hidden-Layer'))
            self.model.add(LSTM(units=128, activation='relu', name='First-Hidden-Recurrent-Layer', return_sequences=True))
            self.model.add(Dense(units=96, activation='relu', name='Second-Hidden-Layer'))
            self.model.add(LSTM(units=64, activation='relu', name='Second-Hidden-Recurrent-Layer'))
            self.model.add(Dense(units=32, activation='relu', name='Third-Hidden-Layer'))
            self.model.add(Dense(units=1, activation='linear', name='F-Output-Layer'))

    def compile(self):
        self.model.compile(optimizer='adam',  # an algorithm to be used in backpropagation
                           loss='MeanSquaredError',  # Loss function to be optimized
                           )

    def fit(self):
        self.history = self.model.fit(x=self.X_train,   # input data
                                      y=self.y_train,   # target data
                                      batch_size=12,
                                      epochs=220,
                                      verbose=1
                                      )

    def performance_summary(self):
        self.pred_train = self.model.predict(self.X_train)
        self.pred_test = self.model.predict(self.X_test)
        print('\n')
        self.model.summary()

        print('\n-------------------- Weights and Biases --------------------')
        print("Note, the last parameter in each layer is bias while the rest are weights")
        for layer in self.model.layers:
            print(layer.name)
            for item in layer.get_weights():
                print(item, "\n\n")

        print('\n---------- Evaluation on Training Data ----------')
        print("MSE: ", mean_squared_error(self.y_train, self.pred_train))

        print('\n---------- Evaluation on Test Data ----------')
        print("MSE: ", mean_squared_error(self.y_test, self.pred_test))

    def plot_loss(self, symbol):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(self.history.history['loss'])
        ax.set_title(f'{symbol} - Model Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.show()


def train_model():
    con = sqlite3.connect("database.db")
    tmp = con.execute(f"SELECT symbol FROM symbols").fetchall()
    con.close()

    tmp = tmp[20:25]
    for i, value in enumerate(tmp):
        tmp[i] = value[0]

    for value in tmp:
        try:
            TrainLSTM(value)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    # train_model()
    LSTMPrediction("tsla")
