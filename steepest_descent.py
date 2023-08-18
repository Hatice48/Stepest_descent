import numpy as np

def f(x):
    return 3 + (x[0] - 1.5*x[1])**2 + (x[1] - 2)**2

def grad_f(x):
    return np.array([2*(x[0] - 1.5*x[1]), 2*(x[1] - 2)])

def steepest_descent(x0, alpha, eps):
    x = x0
    fx = f(x)
    iter_count = 0  # iterasyon sayısını sıfırla
    while True:
        iter_count += 1  # her adımda iterasyon sayısını arttır
        grad = grad_f(x)
        x_new = x - alpha*grad
        fx_new = f(x_new)
        if abs(fx_new - fx) < eps:
            break
        x = x_new
        fx = fx_new
    return x_new, fx_new, iter_count  # iterasyon sayısını da döndür

# Örnek kullanım
x0 = np.array([-4.5, -3.5])
alpha = 0.5
eps = 1e-6

x_min, f_min, iter_count = steepest_descent(x0, alpha, eps)

print("Minimum nokta: ", x_min)
print("Minimum değer: ", f_min)
print("Iterasyon sayısı: ", iter_count)