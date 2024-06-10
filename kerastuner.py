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
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

import IPython
import joblib
import keras_tuner as kt
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
    output_dim = 256
    unit_size = 256
    loss = 'mean_squared_error'
    optimizer = Adam(learning_rate=1e-4, clipnorm=1)
    batch_size = 256
    epochs = 1000
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
    input_feature = ['origin_product_no', 'daily_review_count', 'y'] # 여기부분 변경? -> 예전에 할 때 ezra님은 안건드신거 같음
    checkpoint_path = "/home/ailee/ailee-v1-ai/DNN/training/model_checkpoint.h5"
    dropout_rate = 0.0
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    use_gpu = False

# 해당 부분 변경
def model_builder(hp):
    hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32) # step 변경
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
    hp_dropout_rate = hp.Choice('dropout_rate', values = [0.2, 0.3, 0.5])

    input1 = Input(shape=(1,))
    input2 = Input(shape=(1,))
    embedding = Embedding(input_dim=CFG.input_dim, output_dim=CFG.output_dim, input_length=1)(input1)
    flatten = Flatten()(embedding)
    concat = Concatenate()([flatten, input2])
    batch_norm1 = BatchNormalization()(concat)
    dropout1 = Dropout(hp_dropout_rate)(batch_norm1)

    dense1 = Dense(CFG.output_dim, activation='relu')(dropout1)
    batch_norm2 = BatchNormalization()(dense1)
    dropout2 = Dropout(hp_dropout_rate)(batch_norm2)

    dense2 = Dense(hp_units, activation='relu')(dropout2)
    batch_norm3 = BatchNormalization()(dense2)
    dropout3 = Dropout(hp_dropout_rate)(batch_norm3)
    
    dense3 = Dense(hp_units, activation='relu')(dropout3)
    batch_norm4 = BatchNormalization()(dense3)
    dropout4 = Dropout(hp_dropout_rate)(batch_norm4)
    
    dense4 = Dense(hp_units, activation='relu')(dropout4)
    batch_norm5 = BatchNormalization()(dense4)
    dropout5 = Dropout(hp_dropout_rate)(batch_norm5)
    
    dense5 = Dense(hp_units, activation='relu')(dropout5)
    batch_norm6 = BatchNormalization()(dense5)
    dropout6 = Dropout(hp_dropout_rate)(batch_norm6)
    
    output = Dense(1, activation='linear')(dropout6)

    model = keras.models.Model(inputs=[input1, input2], outputs=output)
    model.compile(loss=CFG.loss, optimizer=keras.optimizers.Adam(learning_rate = hp_learning_rate, clipnorm=1), metrics=['mse', 'mae'])

    return model


def init_logger():
    log_dir = '/home/ailee/ailee-v1-ai/DNN/training' # 경로 변경
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

# 해당 부분은 따로 사용하지 않았음(다운 받아진 데이터를 활용)
def download_dataset(split_days=7):
    s = ShopperDataset(ctg=CFG.ctg)

    review = s.get_review()
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
    dataset['y'] = np.round(dataset['purchaseCnt']/60)

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
    now = datetime.now().strftime("%Y-%m-%d")
    id_list = df['origin_product_no'].unique()

    enc = LabelEncoder()
    enc.fit(df['origin_product_no'])
    np.save('/home/ailee/ailee-v1-ai/DNN/id_list.npy', id_list)
    joblib.dump(enc, f'/home/ailee/ailee-v1-ai/DNN/label_encoder_{now}.joblib')

    return enc

# 따로 사용 X
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

# 따로 할용 X
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


# 따로 사용 X
def get_kpi(data):
    r2 = r2_score(data['purchaseCnt'], data['predict_sales'])
    rmse = np.sqrt(mean_squared_error(data['purchaseCnt'], data['predict_sales']))
    mae = mean_absolute_error(data['purchaseCnt'], data['predict_sales'])
    mape = mean_absolute_percentage_error(data['purchaseCnt'], data['predict_sales'])
    mpe = np.mean((data['purchaseCnt']-data['predict_sales'])/data['purchaseCnt'])

    return pd.Series(dict(r2 = r2, rmse = rmse, mae = mae, mape = mape, mpe = mpe))
    

# 따로 사용 X
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
    train_dataset = pd.read_parquet("/home/ailee/ailee-v1-ai/DNN/data/no_out_log_new_train_20240417.parquet") # train, test set에 따라서 변경
    test_dataset = pd.read_parquet("/home/ailee/ailee-v1-ai/DNN/data/no_out_log_new_test_20240417.parquet")
    LOGGER.info('%-30s | [DONE]' % 'DATA DOWNLOAD')
    train_dataset = dataset_sampling(train_dataset)
    enc = id_encoding(train_dataset)
    test_dataset = test_dataset[test_dataset.origin_product_no.isin(train_dataset.origin_product_no.unique())]
    train_dataset['origin_product_no'] = enc.transform(train_dataset['origin_product_no'])
    test_dataset['origin_product_no'] = enc.transform(test_dataset['origin_product_no'])
    LOGGER.info('%-30s | [START]' % 'BUILD MODEL')

    tuner = kt.Hyperband(model_builder,
                     objective = 'val_mse', 
                     max_epochs = 100,
                     factor = 3,
                     directory = 'training',
                     project_name = 'kt3') # 이 부분 변경
    
    class ClearTrainingOutput(tf.keras.callbacks.Callback):
        def on_train_end(*args, **kwargs):
            IPython.display.clear_output(wait = True)

    X1_train = train_dataset['origin_product_no'].values
    X2_train = train_dataset['log_daily_review_count'].values # 해당 부분도 변경(훈련데이터의 x, y에 맞게 변경)
    y_train = train_dataset['log_y'].values

    X1_test = train_dataset['origin_product_no'].values
    X2_test = train_dataset['log_daily_review_count'].values
    y_test = train_dataset['log_y'].values

    tuner.search([X1_train, X2_train], y_train, epochs = 100, validation_data = ([X1_test, X2_test], y_test), callbacks = [ClearTrainingOutput()])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

    LOGGER.info(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')} and the optimal dropout rate is {best_hps.get('dropout_rate')}.
    """)


if __name__ == '__main__':
    main()