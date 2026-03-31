# import matplotlib.pyplot as plt

# classes = ["Healthy", "Bacterial Spot", "Late Blight", "YLCV"]

# train = [1113, 1488, 1336, 2246]
# val   = [238, 319, 286, 481]
# test  = [240, 320, 287, 482]

# x = range(len(classes))

# plt.figure()
# plt.bar(x, train, label='Train')
# plt.bar(x, val, bottom=train, label='Validation')
# plt.bar(x, test, bottom=[i+j for i,j in zip(train,val)], label='Test')

# plt.xticks(x, classes)
# plt.ylabel("Number of Images")
# plt.title("Dataset Distribution by Class")
# plt.legend()
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# models = [
#     "ResNet50",
#     "EfficientNetB0",
#     "MobileNetV3Small",
#     "EffNetB0+CBAM",
#     "MobileNetV3+SE",
#     "MobileNetV3+CBAM"
# ]

# accuracy = [0.9962, 0.9910, 0.9707, 0.9880, 0.9872, 0.9804]
# f1       = [0.9958, 0.9906, 0.9687, 0.9875, 0.9865, 0.9805]

# x = np.arange(len(models))
# width = 0.35

# plt.figure()
# plt.bar(x - width/2, accuracy, width, label='Accuracy')
# plt.bar(x + width/2, f1, width, label='F1-score')

# plt.xticks(x, models, rotation=30)
# plt.ylabel("Score")
# plt.title("Model Performance Comparison")

# # 🔥 important line (zoom)
# plt.ylim(0.95, 1.0)

# plt.legend()
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

severity = ["Mild", "Moderate", "Severe"]
accuracy = [0.9859, 0.9967, 0.9593]
f1 = [0.7437, 0.7489, 0.7343]

x = np.arange(len(severity))
width = 0.35

plt.figure()
plt.bar(x - width/2, accuracy, width, label='Accuracy')
plt.bar(x + width/2, f1, width, label='Macro F1')

plt.xticks(x, severity)
plt.ylabel("Score")
plt.title("Severity-wise Classification Performance")

# 🔥 zoom like before
plt.ylim(0, 1.0)

plt.legend()
plt.tight_layout()
plt.show()


