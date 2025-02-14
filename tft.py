import random
import pickle
import os
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import RandomSampler, DataLoader
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MultiHorizonMetric, MAPE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, Callback
import matplotlib.pyplot as plt

OPTUNA = False
TRAINING = False
RESUME = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
train_model_path = 'checkpoints/tft-epoch=09-val_loss=2.0939.ckpt'
test_model_path = 'models/tft-epoch=24-val_loss=2.1923.ckpt'
input_csv_path = 'datasets/TSLA-1h.csv'
input_time_steps = 168
output_time_steps = 24
time_weight = 0.95
graph_num = 5
news_vector_dim = 0
directional_penalty = 1
batch_size = 16
num_workers = 4
val_batch_size = batch_size
test_batch_size = batch_size * 5
# stocks = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'TSLA', 'META']
stocks = ['TSLA']
news_vector_cols = [f"news_vector_{i}" for i in range(news_vector_dim)]

class TimeWeightedDirectionalMPSE(MultiHorizonMetric):
    def __init__(self, reduction="mean", time_weight_decay=0.95, directional_penalty=1.5, **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.time_weight_decay = time_weight_decay
        self.directional_penalty = directional_penalty

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_pred = self.to_prediction(y_pred)
        percentage_error = (y_pred - target) / target.clamp(min=1e-6) * 100
        time_steps = y_pred.size(1)
        time_weights = torch.tensor(
            [self.time_weight_decay ** t for t in range(time_steps)], device=y_pred.device
        ).unsqueeze(0)
        weighted_loss = percentage_error.pow(2) * time_weights
        actual_change = target[:, 1:] - target[:, :-1]
        predicted_change = y_pred[:, 1:] - y_pred[:, :-1]
        wrong_direction = (actual_change * predicted_change < 0).float() * self.directional_penalty
        correct_direction = (actual_change * predicted_change >= 0).float()
        directional_penalty = torch.cat([torch.ones_like(actual_change[:, :1]), wrong_direction + correct_direction], dim=1)
        total_loss = (weighted_loss * directional_penalty)
        return total_loss

class TemporalFusionTransformerWithLrScheduler(TemporalFusionTransformer):
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3),
            "monitor": "train_loss",
            "interval": "epoch",
            "frequency": 1
        }
        return [optimizer], [scheduler]

def prepare_data():
    dtypes = {
        'time_idx': 'int32',
        'hour': 'category',
        'close': 'float32',
        'high': 'float32',
        'low': 'float32',
        'volume': 'float32',
        'symbol': 'category',
        'day_of_week': 'category',
        'timezone': 'category',
        'news_vector': 'str'
    }
    df = pd.read_csv(input_csv_path, dtype=dtypes)
    if news_vector_dim:
        news_vectors = pd.DataFrame(
            df["news_vector"].apply(eval).tolist(),
            columns=news_vector_cols,
            dtype='float32'
        )
        df = pd.concat([df.drop(columns=["news_vector", "datetime", "open"]), news_vectors], axis=1)

    normalizer = GroupNormalizer(groups=['symbol'], transformation='softplus')
    for col in ['high', 'low', 'volume']:
        df[col] = normalizer.fit_transform(df[col], df[['symbol']])

    return df


def create_datasets(df, training_cutoff, testing_cutoff):
    training = TimeSeriesDataSet(
        df[df['time_idx'] <= training_cutoff],
        time_idx='time_idx',
        target='close',
        group_ids=['symbol'],
        max_encoder_length=input_time_steps,
        min_prediction_length=output_time_steps,
        max_prediction_length=output_time_steps,
        static_categoricals=['symbol'],
        time_varying_known_reals=['time_idx', 'hour'],
        time_varying_known_categoricals=['day_of_week', 'timezone'],
        time_varying_unknown_reals=['close', 'low', 'high', 'volume'] + news_vector_cols,
        target_normalizer=GroupNormalizer(groups=['symbol'], transformation='softplus'),
        add_relative_time_idx=True,
        add_target_scales=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, df[df['time_idx'] <= testing_cutoff], min_prediction_idx=training_cutoff + 1, stop_randomization=True)
    testing = {}
    for stock in stocks:
        df_stock = df[df['symbol'] == stock]
        testing[stock] = TimeSeriesDataSet.from_dataset(training, df_stock, min_prediction_idx=testing_cutoff + 1, stop_randomization=True)
    return training, validation, testing

def create_trainer():
    early_stopper = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=15, verbose=True, mode="min")
    trainer = Trainer(
        max_epochs=100,
        accelerator='gpu' if DEVICE != 'cpu' else DEVICE,
        gradient_clip_val=1.0,
        precision='bf16-mixed',
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="tft-{epoch:02d}-{val_loss:.4f}",
                save_top_k=1,
                save_last=True,
                monitor="val_loss",
                mode="min"
            ),
            early_stopper
        ],
        enable_progress_bar=True
    )
    return trainer

def test(best_tft, test_dataloader, graph_dataloader):
    best_tft.eval()
    for stock in stocks:
        predictions = best_tft.predict(test_dataloader[stock], return_y=True)
        mape_values = []
        for t in range(output_time_steps):
            mape_t = MAPE().to(DEVICE)(predictions.output[:, t].unsqueeze(1), predictions.y[0][:, t].unsqueeze(1)).to('cpu')
            mape_values.append(mape_t.item() * 100)

        plt.figure(figsize=(10, 6))
        plt.plot(range(output_time_steps), mape_values)
        plt.xlabel("Timestep (each hour)")
        plt.ylabel("MAPE (%)")
        plt.title(f"{stock} Mean Absolute Percentage Error")
        plt.grid()
        plt.show()

        overall_mape = MAPE()(predictions.output, predictions.y[0]) * 100
        first_timestep_mape = MAPE()(predictions.output[:, 0].unsqueeze(1), predictions.y[0][:, 0].unsqueeze(1)) * 100
        print(f"{stock} Overall MAPE: {overall_mape:.2f}%")
        print(f"{stock} First Timestep MAPE: {first_timestep_mape:.2f}%")

        raw_predictions = best_tft.predict(graph_dataloader[stock], return_x=True, mode="raw")
        for idx in range(graph_num):
            best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=False)
        plt.show()

def optuna(train_dataloader, val_dataloader):
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_test",
        n_trials=200,
        max_epochs=30,
        gradient_clip_val_range=(0.25, 0.5),
        hidden_size_range=(64, 256),
        hidden_continuous_size_range=(64, 256),
        attention_head_size_range=(2, 4),
        learning_rate_range=(0.00001, 0.0005),
        dropout_range=(0.0, 0.2),
        trainer_kwargs=dict(
            limit_train_batches=len(train_dataloader) * 0.2,
            accelerator='gpu' if DEVICE != 'cpu' else DEVICE,
            precision='bf16-mixed',
            enable_progress_bar=True
        ),
        reduce_on_plateau_patience=3
    )
    with open("test_study.pkl", "wb") as fout:
        pickle.dump(study, fout)
    print(study.best_trial.params)

def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

    df = prepare_data()
    split_size = (output_time_steps + (5000 // val_batch_size) * val_batch_size) - 1
    testing_cutoff = df['time_idx'].max() - split_size
    training_cutoff = testing_cutoff - split_size
    training, validation, testing = create_datasets(df, training_cutoff, testing_cutoff)

    train_dataloader = training.to_dataloader(batch_size=batch_size, pin_memory=True, num_workers=num_workers, persistent_workers=True, shuffle=True)
    val_dataloader = validation.to_dataloader(batch_size=val_batch_size, pin_memory=True, num_workers=num_workers, persistent_workers=True, shuffle=False)
    test_dataloader = {}
    graph_dataloader = {}
    for stock in stocks:
        test_dataloader[stock] = testing[stock].to_dataloader(batch_size=test_batch_size, pin_memory=True, num_workers=num_workers, persistent_workers=True, shuffle=False)
        graph_dataloader[stock] = testing[stock].to_dataloader(batch_size=1, shuffle=False, sampler=RandomSampler(testing[stock], num_samples=5, replacement=True))


    if OPTUNA:
        optuna(train_dataloader, val_dataloader)
    else:
        loss = TimeWeightedDirectionalMPSE(time_weight_decay=time_weight, directional_penalty=directional_penalty)
        tft = TemporalFusionTransformerWithLrScheduler.from_dataset(
            training,
            learning_rate=0.00001,
            hidden_size=256,
            attention_head_size=4,
            dropout=0.0,
            hidden_continuous_size=256,
            loss=loss,
        )
        if TRAINING:
            trainer = create_trainer()
            trainer.fit(
                tft,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=train_model_path if RESUME else None
            )
            best_model_path = trainer.checkpoint_callback.best_model_path
            best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
            torch.save(best_tft, "models/tft_model_TSLA.pth")
        else:
            if test_model_path.endswith('.ckpt'):
                best_tft = TemporalFusionTransformerWithLrScheduler.load_from_checkpoint(test_model_path).to(DEVICE)
            elif test_model_path.endswith('.pth'):
                model_state_dict = torch.load(test_model_path, weights_only=True)
                best_tft = TemporalFusionTransformerWithLrScheduler().load_state_dict(model_state_dict).to(DEVICE)
            else:
                raise ValueError(f"Unsupported file extension: {os.path.splitext(test_model_path)[-1]}.")
        test(best_tft, test_dataloader, graph_dataloader)

if __name__ == '__main__':
    main()
