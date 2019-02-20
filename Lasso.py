import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


########### Lasso without resampling###############

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2) 
    return term1 + term2 + term3 + term4 

#Introducing noise
sigma = 1
z = FrankeFunction(x, y)
znew = z + 0.01*np.random.normal(0,sigma,(20,20)) # z with noise

#Design matrix
poly=PolynomialFeatures(degree=5) 
X=poly.fit_transform(np.c_[x.ravel(), y.ravel()])

#Optimal regularization strength
lamb= 10**(-8)

#Scikit-learn's lasso module
model = Lasso(alpha=lamb)
model.fit(X, znew.ravel())

zpredict = model.predict(X)

#MSE and R2 
print("MSE (without resampling):", mean_squared_error(znew.ravel(), zpredict))
print("R2 score (without resampling):", r2_score(znew.ravel(), zpredict))         

#######################Lasso with cross-validation#####################

#Regularization strengths
lambdas = np.logspace(-8, 2, 50)

r2_scores = []
CVmse= []

def FrankeFunction(x, y, noise=0):
   term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
   term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
   term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
   term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
   return term1 + term2 + term3 + term4 + noise * np.random.randn(*x.shape)




#Make data
x, y = (a.reshape(-1, 1) for a in np.meshgrid(*np.random.rand(2, 100)))
X = PolynomialFeatures(degree=5, include_bias=False).fit_transform(np.c_[x, y])
z = FrankeFunction(x, y, noise=0.01)

#MSE and R2 for different values of Regularization strenth
for lamb in lambdas:
   model = Lasso(alpha=lamb)
   scores = cross_val_score(model, X, z, cv=5, scoring="r2")
   scoreMSE= cross_val_score(model, X, z, cv=5,scoring="neg_mean_squared_error")
   r2_scores.append(np.mean(scores))
   CVmse.append(np.mean(scoreMSE))
 

#MSE and R2 for optimal regularization strength
print("R2 score (with cross-validation):",np.max(r2_scores))
print("MSE (with cross-validation):",np.min(np.absolute(CVmse)))


#plots figure: R2 versus lambda
plt.semilogx(lambdas, r2_scores, label="R2")
plt.xlabel(r"$\lambda$", fontsize=18)
plt.ylabel(r"$R^2$", fontsize=18)
plt.title(r"$R^2$ score for Lasso {}th order".format(5))
plt.show()
    
#plots figure: MSE versus lambda
plt.semilogx(lambdas,np.absolute(CVmse), label="MSE")
plt.xlabel(r"$\lambda$", fontsize=18)
plt.ylabel(r"$MSE$", fontsize=18)
plt.title(r"$MSE$  for Lasso {}th order".format(5))
plt.show()


