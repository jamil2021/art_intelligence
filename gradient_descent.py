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
    it = []
    costJ = []
    for i in range(iterations):
        # prediction
        fwbx = fwb_x(X, w, b)
        # cost function
        e = fwbx - y
        Jwb = 1/(2*m) * np.dot( e.T, e )
        # Gradient descent step
        if Jwb < 0.01:
            break
        partialJw = 1/m * np.dot (e, X )
        partialJb = 1/m * np.sum(e)

        w = w - (alpha * partialJw)
        b = b - (alpha * partialJb)        

        if i % 3 == 0:
            it.append(i)
            costJ.append(Jwb)
            print("Cost after iteration %d: %f" % (i, Jwb))
    return w,b,it,costJ


# Generate random data for linear regression
np.random.seed(0)
X = 2.5 * np.random.rand(100) + 1.5   
res = 0.2 * np.random.randn(100)
y = 0.3 * X + res

# Perform gradient descent
w = 0.2
b = 0.2
# alpha = 0.005
iterations = 300
w1,b1,it1,costJ1 = gradient_descent(X, y, w, b, 0.05, iterations)
w2,b2,it2,costJ2 = gradient_descent(X, y, w, b, 0.005, iterations)
print(f"w = {w1}, b = {b1}")
# Plotting the results
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue')
plt.plot(X, fwb_x(X, w1, b1), color='red')
plt.title('Gradient Descent Linear Regression (alpha=0.05)')
plt.xlabel('Independent variable')
plt.ylabel('Dependent variable')
plt.show()

plt.plot(it1,costJ1, label='alpha=0.05')
plt.plot(it2,costJ2, label='alpha=0.005')
plt.title('Gradient Descent Linear Regression J')
plt.xlabel('Iterations')
plt.ylabel('Cost J')
plt.legend()
# const_line = [0.01985] * len(it)
# plt.plot(it, const_line, linestyle=':', linewidth=1.0, color='red')
plt.show()


