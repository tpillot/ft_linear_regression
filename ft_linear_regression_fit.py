import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from celluloid import Camera
import sys




def normalize_min_max(x,y):
	max_x = max(x)
	max_y = max(y)
	min_x = min(x)
	min_y = min(y)
	
	x_norm = []
	y_norm = []

	for i in range(len(x)):
		x_norm.append((x[i] - min_x) / (max_x - min_x))
		y_norm.append((y[i] - min_y) / (max_y - min_y))

	return np.array(x_norm),np.array(y_norm)

def model(X, theta):
    return X.dot(theta)
 
def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)
 
def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)
 
def gradient_descent(x, y, theta, learning_rate, n_iterations):
	x_norm, y_norm = normalize_min_max(x,y)
	X_norm = np.hstack((x_norm, np.ones(x_norm.shape)))
	cost_history = np.zeros(n_iterations)
	theta_history = theta.T

	for i in range(0, n_iterations):
		theta = theta - learning_rate * grad(X_norm, y_norm, theta)
		cost_history[i] = cost_function(X_norm, y_norm, theta)
		theta_history = np.append(theta_history,inverse_transforme(theta,x,y).T, axis=0)
 	
	return theta, cost_history, theta_history

def inverse_transforme(theta_final,x,y):
	max_x = max(x)
	min_x = min(x)
	max_y = max(y)
	min_y = min(y)
	x_0 = -1 * min_x /(max_x - min_x)
	x_1 = (1 - min_x) /(max_x - min_x)
	y_0 = theta_final[0] * x_0 + theta_final[1]
	y_1 =  theta_final[0] * x_1 + theta_final[1]
	theta1 = y_0 * (max_y - min_y) + min_y
	theta0 = y_1 * (max_y - min_y) + min_y - theta1
	theta_transforme = np.array([theta0,theta1])
	return(theta_transforme)

def visu(x,y,theta_history):
	fig = plt.figure()
	camera = Camera(fig)
	X = np.hstack((x, np.ones(x.shape)))
	
	for theta in theta_history:
		predictions = model(X, theta)
		plt.scatter(x, y, c='b')
		plt.plot(x, predictions, c='r')
		plt.xlim(0,250000)
		plt.ylim(0,10000)
		camera.snap()


	animation = camera.animate(interval=50, blit=True)



def main():
	if len(sys.argv) == 2:
		data_name = sys.argv[1]
	elif len(sys.argv) < 2:
		print("No path file has been included")
	try:
		df = pd.read_csv(data_name)
	except:
		print("No file" + data_name)

	x = df.iloc[:, [0]].to_numpy()
	y = df.iloc[:, [1]].to_numpy()
	X = np.hstack((x, np.ones(x.shape)))
	print(np.shape([[1,2]]))
	# #X = np.hstack((x, np.ones(x.shape)))
	# theta = np.random.randn(2, 1)
	# n_iterations = 100
	# learning_rate = 0.5
	# x_norm, y_norm = normalize_min_max(x,y)
	# theta_final, cost_history, theta_history = gradient_descent(x, y, theta, learning_rate, n_iterations)
	# #print(theta_final)
	# theta_final = inverse_transforme(theta_final,x,y)
	# #print(theta_final)

	# # création d'un vecteur prédictions qui contient les prédictions de notre modele final
	# X = np.hstack((x, np.ones(x.shape)))
	# predictions = model(X, theta_final)

	# visu(x,y,theta_history)

	

if	__name__ == '__main__':
	main()











