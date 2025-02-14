import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# LSTM Model Definition
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

# Create sequences for training
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length][0])
    return np.array(xs), np.array(ys)

# Compute monthly, quarterly, and yearly averages
def compute_averages(df):
    df['Monthly_Avg'] = df['Close/Last'].rolling(window=21).mean()
    df['Quarterly_Avg'] = df['Close/Last'].rolling(window=63).mean()
    df['Yearly_Avg'] = df['Close/Last'].rolling(window=252).mean()
    return df

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('C:\\Projects\\Final Project\\datasets\\TSLA.csv')[::-1]

    # Clean 'Close/Last' column
    df['Close/Last'] = df['Close/Last'].replace({'\$': '', ',': ''}, regex=True).astype(float)

    # Compute averages
    df = compute_averages(df)
    # Manual Min-Max scaling for 'Close/Last'
    data_min, data_max = df['Close/Last'].min(), df['Close/Last'].max()
    df[['Close/Last', 'Monthly_Avg', 'Quarterly_Avg', 'Yearly_Avg']] = (df[['Close/Last', 'Monthly_Avg', 'Quarterly_Avg', 'Yearly_Avg']] - data_min) / (data_max - data_min)
    data_with_features = df[['Close/Last', 'Monthly_Avg', 'Quarterly_Avg', 'Yearly_Avg']].to_numpy()

    sequence_length = 21
    output_sequence_length = 7
    batch_size = 256
    input_size = 4
    hidden_size = 64
    num_layers = 2
    output_size = 1
    learning_rate = 0.01
    num_epochs = 400

    X, y = create_sequences(data_with_features[252:], sequence_length)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=37, shuffle=False)

    # Data loader for training on either GPU or CPU
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train.to(device), y_train.to(device)), batch_size=batch_size, shuffle=True)

    # Model to GPU/CPU based on availability for training
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # Training the model
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), 'lstm_model.pth')
    print('Model saved as lstm_model.pth')

    # Switch model and data to CPU for testing
    model.cpu()

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=1, shuffle=False)

    # Evaluate the model on CPU
    model.eval()
    mape = np.zeros(output_sequence_length)
    test_predictions, test_gt = [], []
    test_limit = len(test_loader) - 7

    with torch.no_grad():
        for i in range(test_limit):
            input, target = test_loader.dataset[i]
            week_output, week_gt = [], []
            for j in range(output_sequence_length):
                day_output = model(input.unsqueeze(0)).squeeze(0)
                week_output.append(day_output.item())
                week_gt.append(target.item())
                next_day_ma = (input[-1][1].item() * 21 - X[-(29 + 21 - j - i)][-1][1].item() + day_output.item()) / 21
                next_day_qa = (input[-1][2].item() * 63 - X[-(29 + 63 - j - i)][-1][2].item() + day_output.item()) / 63
                next_day_ya = (input[-1][3].item() * 252 - X[-(29 + 252 - j - i)][-1][3].item() + day_output.item()) / 252
                input = torch.cat((input[1:], torch.tensor([day_output.item(), next_day_ma, next_day_qa, next_day_ya]).unsqueeze(0)), dim=0)
                target = test_loader.dataset[i + j + 1][1]

            week_output_original = np.array(week_output) * (data_max - data_min) + data_min
            week_gt_original = np.array(week_gt) * (data_max - data_min) + data_min
            test_predictions.append(week_output_original)
            test_gt.append(week_gt_original)
            mape += np.abs(week_output_original - week_gt_original) / week_gt_original * 100

    mape /= test_limit
    print(f'MAPE for each of the 7 days: {[f"{value:.4f}%" for value in mape]}')

    # Plot the predicted vs actual stock prices for each of the 7 days
    for day in range(1):
        plt.figure(figsize=(10, 5))
        predicted_values = np.array([pred[day] for pred in test_predictions]).flatten()
        actual_values = np.array([gt[day] for gt in test_gt]).flatten()

        plt.plot(range(len(predicted_values)), predicted_values, label='Predicted', linestyle='-', linewidth=1,
                 marker='o')
        plt.plot(range(len(actual_values)), actual_values, label='Actual', linestyle='-', linewidth=1, marker='x')

        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.title(f'Predicted vs Actual for day {day + 1}')
        plt.xticks(range(len(predicted_values)), [f'{i + 1}' for i in range(len(predicted_values))])
        plt.legend()
        plt.grid()

        plt.text(0.02, 0.95, f'Mean Absolute Percentage Error (MAPE): {mape[0]:.4f}%', transform=plt.gca().transAxes,
                 fontsize=12,
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

        # Save the plot as an image
        plt.savefig(f'day_{day + 1}.png')  # Save the image file
        plt.show()  # Display the plot
