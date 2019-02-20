# -*- coding: utf-8 -*-
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

# Load the terrain data
terrain1 = imread('SRTM_data_Norway_1.tif')

# Show the terrain
plt.figure()
plt.title('Terrain over Stavanger, Norway')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.show()



y = np.arange(0, len(terrain1))
x = np.arange(0, len(terrain1[0]))
x, y = np.meshgrid(x,y)

#Plotting the 3D figure
ter = plt.figure()
ax = ter.gca(projection='3d')
surf = ax.plot_surface(x, y, terrain1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

#Customize the axes.
plt.xlabel('X')
ax.xaxis.set_major_locator(LinearLocator(4))
plt.ylabel('Y')
ax.yaxis.set_major_locator(LinearLocator(4))
ax.set_zlim(0, 1900)
ax.set_zticks(np.arange(0,2375,475,dtype=int))
ax.set_title("Stavanger Terrain Height")
ax.pbaspect = [1., .33, 0.5]
ax.view_init(elev=35., azim=-70)   
ax.yaxis.set_rotate_label(False)
ax.yaxis.label.set_rotation(0)
ax.zaxis.set_rotate_label(False)
ax.zaxis.label.set_rotation(0)
ax.dist = 10.5
ter.colorbar(surf, shrink=0.6, aspect=10)
plt.show()


#Now we perform a OLS regression and evaluate the
#corresponding accuracy scores 

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import cross_val_score

poly = PolynomialFeatures(degree=10) #degree 10 gives the lowest MSE
X = poly.fit_transform(np.c_[x.ravel(), y.ravel()]) #Design matrix
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(terrain1.ravel()) #Solution for beta

heightfit = X.dot(beta).reshape(len(terrain1),len(terrain1[0]))

#plot of the fitting polynomial
terfit = plt.figure()
axf = terfit.gca(projection='3d')
surfit = axf.plot_surface(x, y, heightfit, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.xlabel('X')
axf.xaxis.set_major_locator(LinearLocator(4))
plt.ylabel('Y')
axf.yaxis.set_major_locator(LinearLocator(4))
axf.set_zlim(0, 1900)
axf.set_zticks(np.arange(0,2375,475,dtype=int))
axf.set_title("Stavanger Terrain Height Degree-10 Fit")
axf.pbaspect = [1., .33, 0.5]
axf.view_init(elev=35., azim=-70)   
axf.yaxis.set_rotate_label(False)
axf.yaxis.label.set_rotation(0)
axf.zaxis.set_rotate_label(False)
axf.zaxis.label.set_rotation(0)
axf.dist = 10.5
terfit.colorbar(surfit, shrink=0.6, aspect=10)
plt.show()

#MSE and R2 scores
mse=mean_squared_error(terrain1, heightfit)
print("MSE, OLS (without resampling)",mse)
r2=r2_score(terrain1, heightfit)
print("R2 score, OLS (without resampling)",r2)

#######################Bootstrap for OLS ###########################

from sklearn.model_selection import train_test_split

n_bootstraps=10

dim = (X.shape[1], n_bootstraps)
bootmse=np.zeros(n_bootstraps)
bootr2=np.zeros(n_bootstraps)

for i in range(n_bootstraps):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    X_train = poly.fit_transform(np.c_[x_train, y_train])
    X_test = poly.fit_transform(np.c_[x_test, y_test])
    z_train = terrain1[x_train, y_train]
    z_test = terrain1[x_test, y_test]
    betatrain=np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train) 
    zpredictrain = X_train.dot(betatrain)
    zpredictest = X_test.dot(betatrain)
    
    #MSE and R2 scores
    bootmse[i]=mean_squared_error(terrain1.ravel(), zpredictest)
    bootr2[i]=r2_score(terrain1.ravel(), zpredictest)

mseboot=np.mean(bootmse)
r2boot=np.mean(bootr2)

print("MSE, OLS (bootstrap)", mseboot)
print("R2 score, OLS (bootstrap)", r2boot)


###### Ridge without resampling###############

lambd = [0.00001, 0.0001, 0.001, 0.01, 0.1]

for lam in lambd:
    betaRidge = np.linalg.inv(X.T.dot(X)+ lam*np.identity(X.shape[1])).dot(X.T).dot(terrain1.ravel())
    hfitRidge=X.dot(betaRidge).reshape(len(terrain1),len(terrain1[0]))
    r2Ridge=r2_score(terrain1, hfitRidge)
    mseRidge=mean_squared_error(terrain1, hfitRidge)
    
    #MSE and #R2
    print("R2 score, Ridge (without resampling):", r2Ridge)
    print("MSE for Ridge regression (without resampling):", mseRidge)

##### Ridge + Cross-Validation##########################

r2s = np.zeros(len(lambd))
mses = np.zeros(len(lambd))
i = 0

for lamb in lambd:
   model = linear_model.Ridge(alpha=lamb)
   scoreR2 = cross_val_score(model, X, terrain1.ravel(), cv=5, scoring="r2")
   scoreMSE = cross_val_score(model, X, terrain1.ravel(), cv=5,scoring="neg_mean_squared_error")
   print("R2 scores for Ridge cross-validation regression:", np.mean(scoreR2))
   print("MSE for Ridge cross-validation regression:", np.mean(scoreMSE))
   r2s[i] = np.mean(scoreR2)
   mses[i] = np.mean(scoreMSE)
   i += 1

plt.semilogx(lambd, r2s, label="R2")
plt.xlabel(r"$\lambda$", fontsize=18)
plt.ylabel(r"$R^2$", fontsize=18)
plt.title(r"Cross-validation $R^2$ score for Ridge 10th order regression", pad=20)
plt.show()


################ Lasso#######################

lambdas = np.logspace(-8, 2, 50)
lasscore = np.zeros(len(lambdas))
for lamb in lambdas:
  model = linear_model.Lasso(alpha=lamb)
  model.fit(X, terrain1.ravel())
  lasscore = model.score(X, terrain1.ravel())

plt.semilogx(lambdas, lasscore, label="R2")
plt.xlabel(r"$\lambda$", fontsize=18)
plt.ylabel(r"$R^2$", fontsize=18)
plt.title(r"Lasso 10th order regression", pad=20)
plt.show()
