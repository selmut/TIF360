from keras.losses import MeanAbsoluteError

y_true = [[0., 1.], [0., 0.]]
y_pred = [[1., 1.], [1., 0.]]
# Using 'auto'/'sum_over_batch_size' reduction type.
mae = MeanAbsoluteError()
loss = mae(y_true, y_pred).numpy()

print(loss)
