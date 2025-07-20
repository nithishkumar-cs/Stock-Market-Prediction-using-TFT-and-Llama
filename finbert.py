import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel


class FinBERT:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = BertModel.from_pretrained('yiyanghkust/finbert-tone').to(self.device)

    def get_sentiment_vectors(self, text):
        if not isinstance(text, str):
            return np.zeros((1, 768))
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(
            self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()


if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('datasets/SP500-testing.csv', low_memory=False)

    # Extract TSLA columns
    tsla_columns = ['TSLA_close', 'TSLA_high', 'TSLA_low', 'TSLA_open', 'TSLA_volume', 'TSLA_news']
    df_tsla = df[tsla_columns]

    # Initialize FinBERT
    finbert = FinBERT()

    # Get sentiment vectors for the news
    sentiment_vectors = np.array([finbert.get_sentiment_vectors(news) for news in df_tsla['TSLA_news']])

    # Remove the TSLA_news column from the original DataFrame
    df_tsla = df_tsla.drop(columns=['TSLA_news'])

    sentiment_vectors = np.squeeze(sentiment_vectors)

    # Concatenate sentiment vectors to the original dataset
    sentiment_df = pd.DataFrame(sentiment_vectors,
                                columns=[f'sentiment_vector_{i}' for i in range(sentiment_vectors.shape[1])])
    result_df = pd.concat([df_tsla.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)

    # Save to new CSV
    result_df.to_csv('datasets/SP500-testing-with-sentiment.csv', index=False)
    print('FinBERT inference completed. Sentiment vectors saved to SP500-training-with-sentiment.csv.')
