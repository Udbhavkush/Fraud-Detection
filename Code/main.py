import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from keras.layers import LSTM, Dense, Embedding, Dropout,Input, Attention, Layer, Concatenate, Permute, Dot, Multiply, Flatten
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras import backend as K, regularizers, Model, metrics
from sklearn.metrics import recall_score, classification_report, confusion_matrix
import numpy as np


CUR_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CUR_DIR)
DATA_DIR = PARENT_DIR + os.sep + 'Dataset'

train_path = DATA_DIR + os.sep + 'fraudTrain.csv'
test_path = DATA_DIR + os.sep + 'fraudTest.csv'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print(df_train.head())
print(df_train.columns)

feature_encoder = LabelEncoder()

print(df_train['is_fraud'].value_counts())


labels=["Genuine", "Fraud"]

fraud_or_not = df_train["is_fraud"].value_counts().tolist()
values = [fraud_or_not[0], fraud_or_not[1]]
fig, ax = plt.subplots()
ax.pie(values, labels=labels , autopct='%1.1f%%')
plt.title('Genuine and Fraud Percentage Pie Chart')
plt.show()

print('Are there any duplicate values in the dataframe?')
print(df_train.duplicated().any())
print(df_train.dtypes)

df_train["trans_date_trans_time"] = pd.to_datetime(df_train["trans_date_trans_time"])
df_train["dob"] = pd.to_datetime(df_train["dob"])
df_train.drop(columns=['Unnamed: 0','cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'],inplace=True)
df_train.dropna(inplace=True)

df_test["trans_date_trans_time"] = pd.to_datetime(df_test["trans_date_trans_time"])
df_test["dob"] = pd.to_datetime(df_test["dob"])
df_test.drop(columns=['Unnamed: 0','cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'],inplace=True)
df_test.dropna(inplace=True)

def label_encode(df, type="train"):

    #feature_encoder = LabelEncoder()
    if type == "train":
        df["merchant"] = feature_encoder.fit_transform(df["merchant"])
        df["category"] = feature_encoder.fit_transform(df["category"])
        df["gender"] = feature_encoder.fit_transform(df["gender"])
        df["job"] = feature_encoder.fit_transform(df["job"])
    else:
        df["merchant"] = feature_encoder.fit_transform(df["merchant"])
        df["category"] = feature_encoder.fit_transform(df["category"])
        df["gender"] = feature_encoder.fit_transform(df["gender"])
        df["job"] = feature_encoder.fit_transform(df["job"])
    return df


df_train = label_encode(df_train, type="train")
df_test = label_encode(df_test, type="test")


# Creating function for scaling
def Standard_Scaler(df, col_names):
    features = df[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    df[col_names] = features

    return df


X_train =df_train.drop(columns=['is_fraud'])
X_test =df_test.drop(columns=['is_fraud'])
Y_train = df_train['is_fraud']
Y_test = df_test['is_fraud']
col_names = X_train._get_numeric_data().columns #['Amount']
X_train = Standard_Scaler(X_train, col_names)
X_test = Standard_Scaler(X_test, col_names)

# We are going to ensure that we have the same splits of the data every time.
# We can ensure this by creating a KFold object, kf, and passing cv=kf instead of the more common cv=5.

# kf = StratifiedKFold(n_splits=5, shuffle=False)
# rf = RandomForestClassifier(n_estimators=100, random_state=13)
#
# score = cross_val_score(rf, X_train, Y_train, cv=kf, scoring='recall')
# print("Cross Validation Recall scores are: {}".format(score))
# print("Average Cross Validation Recall score: {}".format(score.mean()))
#
# smote_pipeline = make_pipeline(SMOTE(random_state=42),
#                               RandomForestClassifier(n_estimators=100, random_state=13))
#
# score3 = cross_val_score(smote_pipeline, X_train, Y_train, scoring='recall', cv=kf)
# print("Cross Validation Recall Scores are: {}".format(score3))
# print("Average Cross Validation Recall score: {}".format(score3.mean()))

### Reshape input to be 3D [samples, timesteps, features] (format requis par LSTM)
#X_train =  X_train.values.reshape(new_shape)
train_LSTM_X = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
test_LSTM_X = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

inputs=Input((1,11))
x1=LSTM(50,dropout=0.3,recurrent_dropout=0.2, return_sequences=True)(inputs)
x2=LSTM(50,dropout=0.3,recurrent_dropout=0.2)(x1)
outputs=Dense(1,activation='sigmoid')(x2)
model=Model(inputs,outputs)
import tensorflow
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tensorflow.keras.metrics.Recall()])
history=model.fit(train_LSTM_X, Y_train,epochs=2,batch_size=20000)


clf = IsolationForest(contamination=0.4, random_state=42)
clf.fit(X_train)
y_pred = clf.predict(X_train)
is_anomaly = y_pred == -1
print(f"Number of anomalies detected: {sum(is_anomaly)}")

if Y_train is not None:
    print("\nAnomaly detection vs actual labels:")
    print(f"Total samples: {len(Y_train)}")
    print(f"Anomalies in class 0: {sum(is_anomaly[Y_train == 0])}")
    print(f"Anomalies in class 1: {sum(is_anomaly[Y_train == 1])}")

y_pred_binary = np.where(y_pred == -1, 1, 0)
y_true_binary = np.where(Y_train == 1, 1, 0)
recall = recall_score(y_true_binary, y_pred_binary)
print(f"Recall: {recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_true_binary, y_pred_binary))


y_pred_test = clf.predict(X_test)
y_pred_binary = (y_pred_test == -1).astype(int)
recall = recall_score(Y_test, y_pred_binary)
print(f"Recall: {recall:.4f}")
print("\nClassification Report:")
print(classification_report(Y_test, y_pred_binary))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred_binary)
print("\nConfusion Matrix:")
print(cm)