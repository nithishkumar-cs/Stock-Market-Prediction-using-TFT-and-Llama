import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.dates as mdates


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length, [1, 2]])  # Predicting TSLA_high and TSLA_low
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def denormalize(normalized_data, min_values, max_values):
    min_values = min_values.to_numpy().reshape(1, -1)
    max_values = max_values.to_numpy().reshape(1, -1)
    return normalized_data * (max_values - min_values) + min_values


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_train = pd.read_csv('datasets/SP500-training-with-sentiment.csv')
    tsla_columns = ['TSLA_close', 'TSLA_high', 'TSLA_low', 'TSLA_open', 'TSLA_volume']
    sentiment_columns = [col for col in df_train.columns if 'sentiment_vector' in col]
    df_train_tsla = df_train[tsla_columns + sentiment_columns]

    # Separate features for normalization
    stock_data = df_train_tsla[tsla_columns]
    sentiment_data = df_train_tsla[sentiment_columns]

    # Manual normalization of the stock features only
    min_values = stock_data.min()
    max_values = stock_data.max()
    stock_data_normalized = (stock_data - min_values) / (max_values - min_values)

    # Combine normalized stock data with sentiment data
    df_train_tsla_normalized = pd.concat([stock_data_normalized, sentiment_data], axis=1)

    sequence_length = 216
    batch_size = 16
    input_size = len(df_train_tsla_normalized.columns)  # Number of features
    hidden_size = 32
    num_layers = 2
    output_size = 2  # Predicting TSLA_high and TSLA_low
    learning_rate = 0.01
    num_epochs = 40

    X_train, y_train = create_sequences(df_train_tsla_normalized.to_numpy(), sequence_length)
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)

    df_test = pd.read_csv('datasets/SP500-testing-with-sentiment.csv')
    df_test_tsla = df_test[tsla_columns + sentiment_columns]

    # Normalize the test stock data using the same min and max values
    stock_test_data = df_test_tsla[tsla_columns]
    sentiment_test_data = df_test_tsla[sentiment_columns]

    # Manual normalization of the test stock data
    stock_test_data_normalized = (stock_test_data - min_values) / (max_values - min_values)
    df_test_tsla_normalized = pd.concat([stock_test_data_normalized, sentiment_test_data], axis=1)

    X_test, y_test = create_sequences(df_test_tsla_normalized.to_numpy(), sequence_length)
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size,
                                               shuffle=True)

    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'models/lstm_finbert_model.pth')
    print('Model saved as lstm_finbert_model.pth')

    model.cpu()
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test.cpu(), y_test.cpu()), batch_size=1,
                                              shuffle=False)

    model.eval()
    test_predictions, test_gt = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            test_predictions.append(outputs.numpy())
            test_gt.append(targets.numpy())

    test_predictions = np.array(test_predictions).squeeze()
    test_gt = np.array(test_gt).squeeze()

    test_predictions_denorm = denormalize(test_predictions, min_values[1:3], max_values[1:3])
    test_gt_denorm = denormalize(test_gt, min_values[1:3], max_values[1:3])

    mape = np.mean(np.abs((test_predictions_denorm - test_gt_denorm) / test_gt_denorm)) * 100
    print(f'MAPE: {mape:.4f}%')

    # date_range = pd.date_range(start='2019-08-22', periods=test_gt_denorm.shape[0], freq='h')
    #
    # plt.figure(figsize=(10, 5))
    # plt.plot(date_range[:24], test_gt_denorm[:24, 0], label='Actual TSLA High', linestyle='-')
    # plt.plot(date_range[:24], test_predictions_denorm[:24, 0], label='Predicted TSLA High', linestyle='-')
    # plt.plot(date_range[:24], test_gt_denorm[:24, 1], label='Actual TSLA Low', linestyle='-', alpha=0.7)
    # plt.plot(date_range[:24], test_predictions_denorm[:24, 1], label='Predicted TSLA Low', linestyle='-')
    #
    # plt.gca().xaxis.set_major_locator(mdates.HourLocator())
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    #
    # plt.xlabel('Time (HH:MM)')
    # plt.ylabel('Stock Price')
    # plt.title('Predicted vs Actual Stock Prices for TSLA High and Low (First 24 Hours)')
    # plt.legend()
    # plt.grid()
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()

    daily_predictions = []
    daily_gt = []

    for i in range(0, len(test_predictions_denorm), 24):
        if i + 24 <= len(test_predictions_denorm):
            daily_predictions.append(test_predictions_denorm[i:i + 24])
            daily_gt.append(test_gt_denorm[i:i + 24])

    daily_predictions = np.array(daily_predictions)
    daily_gt = np.array(daily_gt)

    daily_predictions = daily_predictions[3:10]
    daily_gt = daily_gt[3:10]

    days = pd.date_range(start='2019-08-25', periods=7, freq='D')

    # Compute the average of high and low for both actual and predicted
    actual_avg = (daily_gt[:, :, 0] + daily_gt[:, :, 1]) / 2
    predicted_avg = (daily_predictions[:, :, 0] + daily_predictions[:, :, 1]) / 2

    # Save and display the plot
    plt.figure(figsize=(10, 5))

    for i in range(actual_avg.shape[0]):
        plt.plot(pd.date_range(start=days[i], periods=24, freq='h'), actual_avg[i],
                 label='Actual TSLA' if i == 0 else "", color='blue', alpha=0.7, linestyle='-')
        plt.plot(pd.date_range(start=days[i], periods=24, freq='h'), predicted_avg[i],
                 label='Predicted TSLA' if i == 0 else "", color='orange', alpha=0.7, linestyle='-')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())

    # Add the MAPE value as a text annotation on the plot
    plt.text(0.02, 0.95, f'Mean Absolute Percentage Error (MAPE): {mape:.4f}%', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.title('Predicted vs Actual Hourly Stock Prices for TSLA')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot to the results directory
    plt.savefig('results/tsla_avg_predicted_vs_actual_with_mape.png')

    # Show the plot
    plt.show()

