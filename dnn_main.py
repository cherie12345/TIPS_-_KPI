from sklearn.discriminant_analysis import StandardScaler
from ShopperDataset import ShopperDataset
from datetime import datetime, timedelta
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
from keras.callbacks import EarlyStopping, History
from keras.layers import BatchNormalization, Concatenate, Dense, Dropout, Embedding, Flatten, Input
from keras.optimizers import Adam
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tensorflow import keras

import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf


class CFG:
    ctg='pet'
    seed = 42
    input_dim = 2000
    output_dim = 128
    loss = 'mean_absolute_error'
    batch_size = 32 # batch_size 변경
    epochs = 1000
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    input_feature = ['origin_product_no', 'daily_review_count', 'y']
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    checkpoint_path = f"/home/ailee/ailee-v1-ai/DNN/training/model_checkpoint_{current_time}.h5"
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    use_gpu = False
    activation_function = 'leaky_relu'
    
    ### hyperparameter
    unit_size = 448
    dropout_rate = 0.2
    optimizer = Adam(learning_rate=0.001, clipnorm=1, weight_decay=0.004)


class ShopperDNN():
    def __init__(self, input_dim, output_dim, unit_size, loss, optimizer, batch_size, epochs, early_stopping, checkpoint_path, dropout_rate):
        super(ShopperDNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.unit_size = unit_size
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.early_stopping = early_stopping
        self.dropout_rate = dropout_rate
        tf.debugging.set_log_device_placement(True)


    def build_model(self):
        input1 = Input(shape=(1,))
        input2 = Input(shape=(1,))
        embedding = Embedding(input_dim=self.input_dim, output_dim=self.output_dim, input_length=1)(input1)
        flatten = Flatten()(embedding)
        concat = Concatenate()([flatten, input2])
        batch_norm1 = BatchNormalization()(concat)
        dropout1 = Dropout(self.dropout_rate)(batch_norm1)

        dense1 = Dense(self.output_dim, activation=CFG.activation_function)(dropout1)
        batch_norm2 = BatchNormalization()(dense1)
        dropout2 = Dropout(self.dropout_rate)(batch_norm2)

        dense2 = Dense(self.unit_size, activation=CFG.activation_function)(dropout2)
        batch_norm3 = BatchNormalization()(dense2)
        dropout3 = Dropout(self.dropout_rate)(batch_norm3)
        
        dense3 = Dense(self.unit_size, activation=CFG.activation_function)(dropout3)
        batch_norm4 = BatchNormalization()(dense3)
        dropout4 = Dropout(self.dropout_rate)(batch_norm4)
        
        dense4 = Dense(self.unit_size, activation=CFG.activation_function)(dropout4)
        batch_norm5 = BatchNormalization()(dense4)
        dropout5 = Dropout(self.dropout_rate)(batch_norm5)
        
        dense5 = Dense(self.unit_size, activation=CFG.activation_function)(dropout5)
        batch_norm6 = BatchNormalization()(dense5)
        dropout6 = Dropout(self.dropout_rate)(batch_norm6)
        
        output = Dense(1, activation='linear')(dropout6)

        model = keras.models.Model(inputs=[input1, input2], outputs=output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['mse', 'mae'])

        return model

def init_logger():
    log_dir = '/home/ailee/ailee-v1-ai/DNN/training'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if(len(logger.handlers) == 3):
        return logger
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler1 = logging.StreamHandler()
    handler1.setLevel(logging.INFO)
    handler1.setFormatter(formatter)

    handler2 = RotatingFileHandler(f'{log_dir}/debug.log', maxBytes=100*1024*1024, backupCount=5)
    handler2.setLevel(logging.DEBUG)
    handler2.setFormatter(formatter)

    handler3 = RotatingFileHandler(f'{log_dir}/error.log', maxBytes=100*1024*1024, backupCount=5)
    handler3.setLevel(logging.ERROR)
    handler3.setFormatter(formatter)

    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.addHandler(handler3)

    return logger


def download_dataset(split_days=7):
    s = ShopperDataset(ctg=CFG.ctg)

    review = s.get_review()
    review.to_parquet("/home/ailee/review_dataset.parquet")
    review, _ = s.create_daterange(review)

    years = [2023, 2023, 2024, 2024, 2024, 2024]
    weeks = ['51', '52', '01', '02', '03', '04']

    combined_info = pd.DataFrame()
    for y, w in zip(years, weeks):
        s.year = y
        s.week = w
        info = s.get_info()
        combined_info = pd.concat([combined_info, info])

    dataset = review.merge(combined_info, on=['origin_product_no', 'create_date'], how='left')
    dataset = dataset[dataset.origin_product_no.isin(review.origin_product_no.unique())].reset_index(drop=True)
    # dataset['y'] = np.round(dataset['purchaseCnt']/60)
    dataset['y'] = dataset['purchaseCnt']

    split_date = dataset['create_date'].max() - timedelta(days=split_days)

    train_dataset = dataset[dataset.create_date < split_date].reset_index(drop=True)[CFG.input_feature]
    test_dataset = dataset[dataset.create_date >= split_date].reset_index(drop=True)[CFG.input_feature]

    train_dataset.to_parquet("/home/ailee/train_dataset.parquet")
    test_dataset.to_parquet("/home/ailee/test_dataset.parquet")

    return train_dataset, test_dataset


def dataset_sampling(df, frac=0.1):
    df_greater_than_zero = df.query("y > 0").reset_index(drop=True)

    df_is_zero = df.query("y == 0").sample(frac=frac, random_state=1).reset_index(drop=True)

    result = pd.concat([df_greater_than_zero, df_is_zero], axis=0)
    
    return result


def id_encoding(df):
    id_list = df['origin_product_no'].unique()

    enc = LabelEncoder()
    enc.fit(df['origin_product_no'])
    np.save('/home/ailee/ailee-v1-ai/DNN/id_list.npy', id_list)
    joblib.dump(enc, f'/home/ailee/ailee-v1-ai/DNN/label_encoder_{CFG.current_time}.joblib')

    return enc

def minmax_x_scaling(df):
    minmax_x = MinMaxScaler()
    minmax_x.fit(df[['daily_review_count']])
    joblib.dump(minmax_x, f'/home/ailee/ailee-v1-ai/DNN/minmax_x_{CFG.current_time}.plk')

    return minmax_x

def minmax_y_scaling(df):
    minmax_y = MinMaxScaler()
    minmax_y.fit(df[['y']])
    joblib.dump(minmax_y, f'/home/ailee/ailee-v1-ai/DNN/minmax_y_{CFG.current_time}.plk')

    return minmax_y

def standard_x_scaling(df):
    standard_x = StandardScaler()
    standard_x.fit(df[['daily_review_count']])
    joblib.dump(standard_x, f'/home/ailee/ailee-v1-ai/DNN/standard_x_{CFG.current_time}.plk')

    return standard_x

def standard_y_scaling(df):
    standard_y = StandardScaler()
    standard_y.fit(df[['y']])
    joblib.dump(standard_y, f'/home/ailee/ailee-v1-ai/DNN/standard_y_{CFG.current_time}.plk')

    return standard_y

def train(model, dataset):
    dataset = dataset.dropna()
    X1 = dataset['origin_product_no'].values
    X2 = dataset['daily_review_count'].values
    y = dataset['y'].values

    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1, X2, y, test_size=0.2)

    history = History()

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CFG.checkpoint_path,
                                                        save_weights_only=False,
                                                        monitor='val_loss',
                                                        mode='min',
                                                        save_best_only=True,
                                                        verbose=1)
    
    model.fit([X1_train, X2_train],y_train,
                epochs=CFG.epochs,
                batch_size=CFG.batch_size,
                validation_data=([X1_val, X2_val], y_val),
                callbacks=[cp_callback, CFG.early_stopping, history])
    
    y_train_pred = model.predict([X1_train, X2_train])
    y_val_pred = model.predict([X1_val, X2_val])

    # best_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    # best_train_mae = mean_absolute_error(y_train, y_train_pred)

    # best_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    # best_val_mae = mean_absolute_error(y_val, y_val_pred)

    return history


def plotting(history, metric='mae'):
    # Access training history
    training_loss = history.history['loss']
    training_metric = history.history[metric]
    validation_loss = history.history['val_loss']
    validation_metric = history.history[f'val_{metric}']

    # Plot training metrics
    plt.figure(figsize=(12, 4))

    # Plot training loss and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training accuracy and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(training_metric, label=f'Training {metric}')
    plt.plot(validation_metric, label=f'Validation {metric}')
    plt.title(f'Training and Validation {metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()

    plt.tight_layout()
    plt.show()


def get_kpi(data):
    r2 = r2_score(data['purchaseCnt'], data['predict_sales'])
    rmse = np.sqrt(mean_squared_error(data['purchaseCnt'], data['predict_sales']))
    mae = mean_absolute_error(data['purchaseCnt'], data['predict_sales'])
    mape = mean_absolute_percentage_error(data['purchaseCnt'], data['predict_sales'])
    mpe = np.mean((data['purchaseCnt']-data['predict_sales'])/data['purchaseCnt'])

    return pd.Series(dict(r2 = r2, rmse = rmse, mae = mae, mape = mape, mpe = mpe))
    

def test(model, dataset):
    dataset = dataset.dropna()
    X1 = dataset['origin_product_no'].values
    X2 = dataset['daily_review_count'].values
    y = dataset['y'].values
    dataset['predict_sales'] = np.round(model.predict([dataset['origin_product_no'], dataset['daily_review_count']])/3)

    dataset.rename({'y': 'purchaseCnt'}, axis=1, inplace=True)

    dataset = pd.DataFrame(get_kpi(dataset.fillna(0)))
    
    return dataset

    
def seed_keras(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    LOGGER = init_logger()
    LOGGER.info('%-30s | [START]' % 'MAIN')
    seed_keras(seed=CFG.seed)
    LOGGER.info('%-30s | [START]' % 'DATA DOWNLOAD')
    # train_dataset, test_dataset = download_dataset()

    ### train & test dataset 최신으로 바꾸기
    train_dataset = pd.read_parquet("/home/ailee/ailee-v1-ai/DNN/data/train_20240604.parquet")
    test_dataset = pd.read_parquet("/home/ailee/ailee-v1-ai/DNN/data/test_20240604.parquet")
    # train_dataset = pd.read_parquet("/home/ailee/ailee-v1-ai/DNN/data/train.parquet")
    # test_dataset = pd.read_parquet("/home/ailee/ailee-v1-ai/DNN/data/test.parquet")
    train_dataset.fillna(0, inplace=True)
    test_dataset.fillna(0, inplace=True)
    # train_dataset = train_dataset[(train_dataset.y>0) & (train_dataset.daily_review_count > 0)]
    # test_dataset = test_dataset[(test_dataset.y>0)]
    LOGGER.info('%-30s | [DONE]' % 'DATA DOWNLOAD')
    train_dataset = dataset_sampling(train_dataset)
    enc = id_encoding(train_dataset)
    test_dataset = test_dataset[test_dataset.origin_product_no.isin(train_dataset.origin_product_no.unique())]
    train_dataset['origin_product_no'] = enc.transform(train_dataset['origin_product_no'])
    test_dataset['origin_product_no'] = enc.transform(test_dataset['origin_product_no'])

    minmax_x = minmax_x_scaling(train_dataset)
    train_dataset['daily_review_count'] = minmax_x.transform(train_dataset[['daily_review_count']])
    test_dataset['daily_review_count'] = minmax_x.transform(test_dataset[['daily_review_count']])
    
    # standard_x = minmax_x_scaling(train_dataset)
    # train_dataset['daily_review_count'] = standard_x.transform(train_dataset[['daily_review_count']])
    # test_dataset['daily_review_count'] = standard_x.transform(test_dataset[['daily_review_count']])

    # standard_y = standard_y_scaling(train_dataset)
    # train_dataset['y'] = standard_y.transform(train_dataset[['y']])
    # test_dataset['y'] = standard_y.transform(test_dataset[['y']])

    LOGGER.info('%-30s | [START]' % 'BUILD MODEL')
    model = ShopperDNN(input_dim= CFG.input_dim,
                       output_dim=CFG.output_dim,
                       unit_size=CFG.unit_size,
                       loss=CFG.loss,
                       optimizer=CFG.optimizer,
                       batch_size=CFG.batch_size,
                       epochs=CFG.epochs,
                       early_stopping=CFG.early_stopping,
                       checkpoint_path=CFG.checkpoint_path,
                       dropout_rate=CFG.dropout_rate).build_model()
    LOGGER.info('%-30s | [DONE]' % 'BUILD MODEL')
    LOGGER.info('%-30s | [START]' % 'TRAIN')
    history = train(model, dataset=train_dataset)
    LOGGER.info('%-30s | [DONE]' % 'TRAIN')
    plotting(history)
    LOGGER.info('%-30s | [START]' % 'LOAD BEST MODEL')
    best_model = keras.models.load_model(CFG.checkpoint_path)
    LOGGER.info('%-30s | [DONE]' % 'LOAD BEST MODEL')
    LOGGER.info('%-30s | [START]' % 'TEST')
    test_metric = test(model, dataset=test_dataset)
    test_metric.plot()
    LOGGER.info('%-30s | [START]' % 'TEST')
    LOGGER.info('%-30s | [DONE]' % 'MAIN')


if __name__ == '__main__':
    main()