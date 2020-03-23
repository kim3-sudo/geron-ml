from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import the MNIST dataset
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