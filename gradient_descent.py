import numpy as np
import matplotlib.pyplot as plt


# Linear Regression Model
# X is vector
# w, b are parameters for the linear regression
def fwb_x(X, w, b):
    return w*X + b

# Gradient descent implementation
def gradient_descent(X, y, w, b, alpha=0.05, iterations=100):
    m = len(X)
    for i in range(iterations):
        # prediction
        fwbx = fwb_x(X, w, b)
        # cost function
        e = fwbx - y
        Jwb = 1/(2*m) * np.dot( e.T, e )
        # Gradient descent step
        if Jwb < 0.02:
            break
        partialJw = 1/m * np.dot (e, X )
        partialJb = 1/m * np.sum(e)

        w = w - (alpha * partialJw)
        b = b - (alpha * partialJb)

        

        if i % 10 == 0:
            print("Cost after iteration %d: %f" % (i, Jwb))

    return w,b


# Generate random data for linear regression
np.random.seed(0)
X = 2.5 * np.random.rand(100) + 1.5   
res = 0.2 * np.random.randn(100)
y = 0.3 * X + res

# Perform gradient descent
w = 0.2
b = 0.2
alpha = 0.005
iterations = 10000
w,b = gradient_descent(X, y, w, b, alpha, iterations)
print(f"w = {w}, b = {b}")
# Plotting the results
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue')
plt.plot(X, fwb_x(X, w, b), color='red')
plt.title('Gradient Descent Linear Regression')
plt.xlabel('Independent variable')
plt.ylabel('Dependent variable')
plt.show()



