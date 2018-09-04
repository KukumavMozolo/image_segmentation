import matplotlib.pylab as plt
import pandas as pd
import numpy as np

path_cross = "model/val_hist_K.binary_crossentropy_50_epochs_{}_trial_lambda_.csv"

path_cor = "model/val_hist_correlation_coefficient_loss_50_epochs_1e-05_{} trial_lambda_.csv"

results_corr = list()
results_cross = list()

data_cor = np.zeros((4, 50))
data_cross = np.zeros((4, 50))

for i in range(4):
    eval_res = pd.read_csv(path_cor.format(i))
    eval_res = eval_res['val_jaccard_coef'].as_matrix()
    n = eval_res.shape
    data_cor[i, 0:n[0]] = eval_res

for i in range(4):
    eval_res = pd.read_csv(path_cross.format(i))
    eval_res = eval_res['val_jaccard_coef'].as_matrix()
    n = eval_res.shape
    data_cross[i, 0:n[0]] = eval_res

mean_results_cross = np.mean(data_cross, axis=0)
mean_results_corr = np.mean(data_cor, axis=0)
var_results_cross = np.var(data_cross, axis=0)
var_results_corr = np.var(data_cor, axis=0)
idx = np.linspace(0, 49, 50)

fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(idx, mean_results_cross, c='b', label="binary_crossentropy")
plt.plot(idx, mean_results_corr, c='r', label="correlation_coefficient_loss")
plt.ylabel("jaccard_coef mean")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(idx, var_results_cross, c='b', label="binary_crossentropy")
plt.plot(idx, var_results_corr, c='r', label="correlation_coefficient_loss")
plt.ylabel("jaccard_coef variance")
plt.xlabel("epoche")
plt.legend()
plt.show()
