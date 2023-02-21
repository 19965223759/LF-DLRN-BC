import pandas as pd
import scipy.stats as st

print('-----------------------SVM----------------------')
#训练集
print('------------------------------------------------------------------test----------------------------------------------------------------------')
sens = pd.DataFrame([0.800,0.800,0.860,0.780,0.800,0.820, 0.880,0.800,0.820,0.780])
spec = pd.DataFrame([0.868,0.853,0.853,0.868,0.860,0.838,0.838,0.875,0.882,0.860])
auc = pd.DataFrame([0.834, 0.826,0.856,0.824,0.830,0.829,0.859,0.838,0.851,0.820])
acc = pd.DataFrame([0.849,0.839,0.855,0.844,0.844,0.833,0.849,0.855,0.866,0.839])
ppv = pd.DataFrame([0.690,0.667,0.683,0.684,0.678,0.651,0.667,0.702,0.719,0.672])
npv = pd.DataFrame([0.922,0.921,0.943,0.915,0.921,0.927,0.950,0.922,0.930,0.914])

# train_sens
mean_sens = sens.mean(axis=0)
nstd_sens = sens.std(axis=0)
tscore = st.t.ppf(1 - 0.001, sens.shape[0] - 1)
lower = mean_sens - tscore * nstd_sens / (sens.shape[0] ** 1/2)
upper = mean_sens + tscore * nstd_sens / (sens.shape[0] ** 1/2)
print('train-SVM-sens-mean:',mean_sens.values)
print("train-SVM-sens-95%CI =[{},{}]".format(lower[0], upper[0]))

# train_spec
mean_spec = spec.mean(axis=0)
nstd_spec = spec.std(axis=0)
tscore = st.t.ppf(1 - 0.001, spec.shape[0] - 1)
lower = mean_spec - tscore * nstd_spec / (spec.shape[0] ** 1/2)
upper = mean_spec + tscore * nstd_spec / (spec.shape[0] ** 1/2)
print('train-SVM-spec-mean:',mean_spec.values)
print("train-SVM-spec-95%CI =[{},{}]".format(lower[0], upper[0]))

# train_auc
mean_auc = auc.mean(axis=0)
nstd_auc = auc.std(axis=0)
tscore = st.t.ppf(1 - 0.001, auc.shape[0] - 1)
lower = mean_auc - tscore * nstd_auc / (auc.shape[0] ** 1/2)
upper = mean_auc + tscore * nstd_auc / (auc.shape[0] ** 1/2)
print('train-SVM-auc-mean:',mean_auc.values)
print("train-SVM-auc-95%CI =[{},{}]".format(lower[0], upper[0]))

# train_acc
mean_acc = acc.mean(axis=0)
nstd_acc  = acc .std(axis=0)
tscore = st.t.ppf(1 - 0.001, acc.shape[0] - 1)
lower = mean_acc - tscore * nstd_acc / (acc.shape[0] ** 1/2)
upper = mean_acc + tscore * nstd_acc / (acc.shape[0] ** 1/2)
print('train-SVM-acc-mean:',mean_acc.values)
print("train-SVM-acc-95%CI =[{},{}]".format(lower[0], upper[0]))

# train_ppv
mean_ppv = ppv.mean(axis=0)
nstd_ppv  = ppv .std(axis=0)
tscore = st.t.ppf(1 - 0.001, ppv.shape[0] - 1)
lower = mean_ppv - tscore * nstd_ppv / (ppv.shape[0] ** 1/2)
upper = mean_ppv + tscore * nstd_ppv / (ppv.shape[0] ** 1/2)
print('train-SVM-ppv-mean:',mean_ppv.values)
print("train-SVM-ppv-95%CI =[{},{}]".format(lower[0], upper[0]))

# train_npv
mean_npv = npv.mean(axis=0)
nstd_npv  = npv .std(axis=0)
tscore = st.t.ppf(1 - 0.001, npv.shape[0] - 1)
lower = mean_npv - tscore * nstd_npv / (npv.shape[0] ** 1/2)
upper = mean_npv + tscore * nstd_npv / (npv.shape[0] ** 1/2)
print('train-SVM-npv-mean:',mean_npv.values)
print("train-SVM-npv-95%CI =[{},{}]".format(lower[0], upper[0]))

