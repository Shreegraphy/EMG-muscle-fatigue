import os
import random
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

data_folder = r'C:\Users\shree\Downloads\Fatigue Data'

def stratified_random_sampling(data_folder):
    exclude_files = ['U1Ex1Rep3.csv', 'U3Ex2Rep3.csv']
    file_names = [filename for filename in os.listdir(data_folder) if filename not in exclude_files]

    strata = [
        'Ex1Rep1', 'Ex1Rep2', 'Ex1Rep3',
        'Ex2Rep1', 'Ex2Rep2', 'Ex2Rep3',
        'Ex3Rep1', 'Ex3Rep2', 'Ex3Rep3'
    ]

    num_files = len(file_names)
    num_train = int(num_files * 0.8)
    n_str_train = num_train // len(strata)

    training_files = [random.sample([file_name for file_name in file_names if stratum in file_name], n_str_train)
                      for stratum in strata]
    training_files = [file_name for sublist in training_files for file_name in sublist]

    testing_files = list(set(file_names) - set(training_files))

    return training_files, testing_files

def clean_emg_file(files, data_folder):
    all_fat_epoch = np.array([])
    all_nonf_epoch = np.array([])

    for filename in files:
        file_path = os.path.join(data_folder, filename)
        EMG = pd.read_csv(file_path, names=['time', 'data', 'label'])
        raw_data = EMG['data']

        Fs = 1926
        Fnyquist = Fs / 2
        lowCut = 20 / Fnyquist
        highCut = 500 / Fnyquist

        b, a = signal.butter(4, [lowCut, highCut], btype='band')
        band_data = signal.filtfilt(b, a, raw_data)

        notch_freq = 50
        f_normalized = notch_freq / Fnyquist
        bw = 0.5
        Q = notch_freq / bw
        b, a = signal.iirnotch(f_normalized, Q, fs=Fs)
        filtered_data = signal.filtfilt(a, b, band_data)

        rec_signal = np.abs(filtered_data - np.mean(filtered_data))
        f_cutoff = 20
        b, a = signal.butter(2, f_cutoff * 1.25 / Fnyquist)
        filtered_rec = signal.filtfilt(b, a, rec_signal)

        EMG['data'] = filtered_rec

        if EMG['label'].iloc[-1] == 0:
            EMG.drop(index=len(EMG) - 1, inplace=True)

        fatigue_indices = np.where(EMG['label'] == 1)[0]
        nonFatigue_indices = np.where(EMG['label'] == 0)[0]

        fatigueData = EMG['data'][fatigue_indices].values
        nonFatigueData = EMG['data'][nonFatigue_indices].values

        n_fatEpochs = len(fatigueData) // Fs
        n_nonfEpochs = len(nonFatigueData) // Fs

        fatEpoch_EMG = np.empty((n_fatEpochs, Fs))
        for ep in range(n_fatEpochs):
            startIdx = ep * Fs
            endIdx = (ep + 1) * Fs
            fatEpoch_EMG[ep] = fatigueData[startIdx:endIdx]

        nonfEpoch_EMG = np.empty((n_nonfEpochs, Fs))
        for ep in range(n_nonfEpochs):
            startIdx = ep * Fs
            endIdx = (ep + 1) * Fs
            nonfEpoch_EMG[ep] = nonFatigueData[startIdx:endIdx]

        if all_fat_epoch.size == 0:
            all_fat_epoch = fatEpoch_EMG
        else:
            all_fat_epoch = np.vstack((all_fat_epoch, fatEpoch_EMG))

        if all_nonf_epoch.size == 0:
            all_nonf_epoch = nonfEpoch_EMG
        else:
            all_nonf_epoch = np.vstack((all_nonf_epoch, nonfEpoch_EMG))

    return all_fat_epoch, all_nonf_epoch

def extract_features_emg(all_fat_epoch, all_nonf_epoch):
    if len(all_fat_epoch.shape) == 1:
        all_fat_epoch = np.expand_dims(all_fat_epoch, axis=0)
    if len(all_nonf_epoch.shape) == 1:
        all_nonf_epoch = np.expand_dims(all_nonf_epoch, axis=0)

    Mav_fat = np.mean(np.abs(all_fat_epoch), axis=1)
    Mav_nonf = np.mean(np.abs(all_nonf_epoch), axis=1)

    Var_fat = np.var(all_fat_epoch, axis=1)
    Var_nonf = np.var(all_nonf_epoch, axis=1)

    Rms_fat = np.sqrt(np.mean(all_fat_epoch**2, axis=1))
    Rms_nonf = np.sqrt(np.mean(all_nonf_epoch**2, axis=1))

    Fs = 1926
    frequencies_f, psd_f = welch(all_fat_epoch, fs=Fs, nperseg=Fs, axis=1)
    frequencies_n, psd_n = welch(all_nonf_epoch, fs=Fs, nperseg=Fs, axis=1)

    Mnf_fat = np.sum(frequencies_f * psd_f, axis=1) / np.sum(psd_f, axis=1)
    Mnf_nonf = np.sum(frequencies_n * psd_n, axis=1) / np.sum(psd_n, axis=1)

    Sc_fat = np.sum(frequencies_f * psd_f, axis=1) / np.sum(psd_f, axis=1)
    Sc_nonf = np.sum(frequencies_n * psd_n, axis=1) / np.sum(psd_n, axis=1)

    feature_matrix_fat = np.array([Mav_fat, Var_fat, Rms_fat, Mnf_fat, Sc_fat]).T
    feature_matrix_nonf = np.array([Mav_nonf, Var_nonf, Rms_nonf, Mnf_nonf, Sc_nonf]).T

    feature_matrix = np.concatenate((feature_matrix_fat, feature_matrix_nonf), axis=0)
    class_labels = np.concatenate((np.zeros(feature_matrix_fat.shape[0]), np.ones(feature_matrix_nonf.shape[0])))

    num_samples = feature_matrix.shape[0]
    random_indices = np.random.permutation(num_samples)
    feature_matrix = feature_matrix[random_indices]
    class_labels = class_labels[random_indices]

    standard_scaler = StandardScaler()
    feature_matrix = standard_scaler.fit_transform(feature_matrix)

    return feature_matrix, class_labels

def rnn_training(dl_feature_matrix_tra, dl_class_vector_tra):
    X_train, X_val, y_train, y_val = train_test_split(dl_feature_matrix_tra, dl_class_vector_tra, test_size=0.2, random_state=42)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    model = Sequential()
    model.add(SimpleRNN(units=32, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[early_stopping])

    val_accuracy = history.history['val_accuracy'][-1]
    print("Validation accuracy of the model:", val_accuracy)

    return model

def rnn_testing(dl_feature_matrix_test, dl_class_vector_test, model):
    dl_feature_matrix_test = dl_feature_matrix_test.reshape(dl_feature_matrix_test.shape[0], dl_feature_matrix_test.shape[1], 1)
    dl_test_prediction = model.predict(dl_feature_matrix_test)
    dl_test_prediction_classes = np.round(dl_test_prediction).astype(int)
    dl_test_true_classes = dl_class_vector_test.astype(int)

    dl_test_accuracy = accuracy_score(dl_test_true_classes, dl_test_prediction_classes)
    print("Test accuracy:", dl_test_accuracy)

    return dl_test_accuracy, dl_test_true_classes, dl_test_prediction_classes

def visualize_results(y_test, y_pred):
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('ROC Curve')
    plt.show()

if __name__ == "__main__":
    training_files, testing_files = stratified_random_sampling(data_folder)
    all_fat_epoch, all_nonf_epoch = clean_emg_file(training_files, data_folder)
    feature_matrix, class_labels = extract_features_emg(all_fat_epoch, all_nonf_epoch)

    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, class_labels, test_size=0.2, random_state=42)

    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)

    y_pred = logistic_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy (Logistic Regression):", test_accuracy)
    visualize_results(y_test, y_pred)

    dl_feature_matrix_tra, dl_class_vector_tra = X_train, y_train
    dl_feature_matrix_test, dl_class_vector_test = X_test, y_test

    rnn_model = rnn_training(dl_feature_matrix_tra, dl_class_vector_tra)
    dl_test_accuracy, dl_test_true_classes, dl_test_prediction_classes = rnn_testing(dl_feature_matrix_test, dl_class_vector_test, rnn_model)
    print("Test Accuracy (RNN):", dl_test_accuracy)
    visualize_results(dl_test_true_classes, dl_test_prediction_classes)
