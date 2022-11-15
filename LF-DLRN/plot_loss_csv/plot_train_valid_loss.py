import pandas as pd
from matplotlib import pyplot as plt

df1 = pd.read_csv('./train_val_loss.csv')
step1 = df1['Epoch'].values.tolist()
loss1 = df1['train_loss'].values.tolist()
loss2 = df1['valid_loss'].values.tolist()

#
# df2 = pd.read_csv('./T11-valLoss.csv')
# step2 = df2['Step'].values.tolist()
# loss2 = df2['loss'].values.tolist()




plt.figure(figsize=(10,7))

plt.plot(step1, loss1, label='Training Loss')
plt.plot(step1, loss2, label='Validation Loss')
plt.xlabel("Epoch",size =12)
plt.ylabel("Loss",size =12)
plt.legend(fontsize=10)
plt.title('Training Loss and Validation Loss on T4 CT images',size=15)
plt.savefig(r'./T4_loss.jpg')
plt.show()

