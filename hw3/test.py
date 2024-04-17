import numpy as np

# 创建一个数组
arr = np.array([1, 2, 3])

# 广播数组到指定形状
broadcasted_arr = np.broadcast_to(arr, (4, 3))
print(broadcasted_arr)