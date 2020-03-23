import matplotlib as mpl
import matplotlib.pyplot as plt

# Import the MNIST dataset
from sklearn.datases import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

# Look at the arrays
X, y = mnist["data"], mnist["target"]
X.shape
y.shape

# Try to plot the picture
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

# What is that supposed to be?
y[0]

# Typecast Y as an int
y = y.astype(np.uint8)

# Separate training and testing data
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Training a binary classifier
y_train_5 = (y_train == 5) # True for all 5s, false for all other numbers
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])
