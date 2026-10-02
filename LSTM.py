
# ============================================================
# MODÈLE LSTM  !!! a cmmenter
# ============================================================

# ============================================================
# LSTM - PRÉDICTION DE LA CONGESTION À +2H
# ============================================================

# import tensorflow as tf

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# # ------------------------------------------------------------
# # 1. Paramètres
# # ------------------------------------------------------------

# SEQUENCE_LENGTH = 8

# # ------------------------------------------------------------
# # 2. Création des séquences
# # ------------------------------------------------------------

# def create_sequences(X, y, sequence_length):

#     X_seq = []
#     y_seq = []

#     X = np.asarray(X)
#     y = np.asarray(y)

#     for i in range(sequence_length, len(X)):

#         X_seq.append(
#             X[i-sequence_length:i]
#         )

#         y_seq.append(
#             y[i]
#         )

#     return np.asarray(X_seq), np.asarray(y_seq)


# # ------------------------------------------------------------
# # 3. Séparation temporelle
# # ------------------------------------------------------------

# # Exemple :
# # 70 % train
# # 15 % validation
# # 15 % test

# n = len(X)

# train_end = int(n * 0.70)
# val_end = int(n * 0.85)

# X_val = X[train_end:val_end]
# y_val = y[train_end:val_end]



# # ------------------------------------------------------------
# # 4. Création des séquences
# # ------------------------------------------------------------

# X_train_seq, y_train_seq = create_sequences(
#     X_train,
#     y_train,
#     SEQUENCE_LENGTH
# )

# # Pour validation :
# # on conserve les dernières observations du train
# X_val_extended = np.concatenate(
#     [
#         X_train[-SEQUENCE_LENGTH:],
#         X_val
#     ],
#     axis=0
# )

# y_val_extended = np.concatenate(
#     [
#         y_train[-SEQUENCE_LENGTH:],
#         y_val
#     ],
#     axis=0
# )

# X_val_seq, y_val_seq = create_sequences(
#     X_val_extended,
#     y_val_extended,
#     SEQUENCE_LENGTH
# )


# # Pour test :
# # on conserve les dernières observations de validation

# X_test_extended = np.concatenate(
#     [
#         X_val_extended[-SEQUENCE_LENGTH:],
#         X_test
#     ],
#     axis=0
# )

# y_test_extended = np.concatenate(
#     [
#         y_val_extended[-SEQUENCE_LENGTH:],
#         y_test
#     ],
#     axis=0
# )

# X_test_seq, y_test_seq = create_sequences(
#     X_test_extended,
#     y_test_extended,
#     SEQUENCE_LENGTH
# )


# # ------------------------------------------------------------
# # 5. Vérification des dimensions
# # ------------------------------------------------------------

# print("\nDimensions LSTM")
# print("-" * 50)

# print("X_train_seq :", X_train_seq.shape)
# print("y_train_seq :", y_train_seq.shape)

# print("X_val_seq   :", X_val_seq.shape)
# print("y_val_seq   :", y_val_seq.shape)

# print("X_test_seq  :", X_test_seq.shape)
# print("y_test_seq  :", y_test_seq.shape)


# # ------------------------------------------------------------
# # 6. Construction du modèle
# # ------------------------------------------------------------

# model_lstm = Sequential([

#     LSTM(
#         64,
#         return_sequences=True,
#         input_shape=(
#             X_train_seq.shape[1],
#             X_train_seq.shape[2]
#         )
#     ),

#     Dropout(0.20),

#     LSTM(
#         32,
#         return_sequences=False
#     ),

#     Dropout(0.20),

#     Dense(
#         16,
#         activation="relu"
#     ),

#     Dense(
#         1,
#         activation="linear"
#     )
# ])


# # ------------------------------------------------------------
# # 7. Compilation
# # ------------------------------------------------------------

# model_lstm.compile(

#     optimizer=tf.keras.optimizers.Adam(
#         learning_rate=0.001
#     ),

#     loss="mse",

#     metrics=["mae"]
# )


# # ------------------------------------------------------------
# # 8. Callbacks
# # ------------------------------------------------------------

# early_stopping = EarlyStopping(

#     monitor="val_loss",

#     patience=10,

#     restore_best_weights=True
# )


# reduce_lr = ReduceLROnPlateau(

#     monitor="val_loss",

#     factor=0.5,

#     patience=5,

#     min_lr=1e-6
# )


# # ------------------------------------------------------------
# # 9. Entraînement
# # ------------------------------------------------------------

# history = model_lstm.fit(

#     X_train_seq,
#     y_train_seq,

#     validation_data=(
#         X_val_seq,
#         y_val_seq
#     ),

#     epochs=100,

#     batch_size=32,

#     callbacks=[
#         early_stopping,
#         reduce_lr
#     ],

#     verbose=1,

#     shuffle=False
# )


# # ------------------------------------------------------------
# # 10. Prédiction
# # ------------------------------------------------------------

# y_pred_lstm = model_lstm.predict(
#     X_test_seq,
#     verbose=0
# ).ravel()


# # ------------------------------------------------------------
# # 11. Évaluation
# # ------------------------------------------------------------

# mae_lstm = mean_absolute_error(
#     y_test_seq,
#     y_pred_lstm
# )

# rmse_lstm = np.sqrt(
#     mean_squared_error(
#         y_test_seq,
#         y_pred_lstm
#     )
# )

# r2_lstm = r2_score(
#     y_test_seq,
#     y_pred_lstm
# )


# # ------------------------------------------------------------
# # 12. Résultats
# # ------------------------------------------------------------

# print("\n" + "=" * 70)
# print("RÉSULTATS LSTM +2H")
# print("=" * 70)

# print(f"MAE  = {mae_lstm:.4f}")
# print(f"RMSE = {rmse_lstm:.4f}")
# print(f"R²   = {r2_lstm:.4f}")

# print("=" * 70)