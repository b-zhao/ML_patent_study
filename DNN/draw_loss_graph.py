import matplotlib.pyplot as plt
import numpy as np

# plot loss curve
RESULT_PATH = './models/2019_11_16_08/'

TRAIN_LOSS_PATH = RESULT_PATH + 'train_loss.npy'
TEST_LOSS_PATH = RESULT_PATH + 'test_loss.npy'

train_loss = np.load(TRAIN_LOSS_PATH)
test_loss = np.load(TEST_LOSS_PATH)

print(train_loss.shape)
print(test_loss.shape)

train_x = np.arange(train_loss.shape[0])
test_x = np.arange(test_loss.shape[0])


plt.title('Loss')
plt.plot(train_x, train_loss, color='green', label='training loss')
plt.plot(test_x, test_loss, color='red', label='testing loss')
# plt.plot(x_axix, train_pn_dis,  color='skyblue', label='PN distance')
# plt.plot(x_axix, thresholds, color='blue', label='threshold')
plt.legend() # 显示图例

plt.xlabel('iterations')
plt.ylabel('loss value')
plt.savefig('./loss.jpg')
plt.show()
