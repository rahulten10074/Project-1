import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score,make_scorer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

########### Ridge without resampling###############

# Make data.
npoints = 20
x = np.linspace(0, 1, npoints)
y = np.linspace(0, 1, npoints)
x, y = np.meshgrid(x,y)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 
    
#Design matrix  
poly=PolynomialFeatures(degree=5)
X=poly.fit_transform(np.c_[x.ravel(), y.ravel()])

#Introducing noise noise
z = FrankeFunction(x, y)
sigma=0.01
znew = z + np.random.normal(0,sigma,(npoints,npoints))


#Regularization parameter that gives best result.
lambd=0.00001

#Solution for betas
betaRid = np.linalg.inv(X.T.dot(X)+lambd*np.identity(X.shape[1])).dot(X.T).dot(znew.ravel())

#Computing Validation scores
zpredridge=X.dot(betaRid).reshape(20,20)
print("R2 score (without resampling):",r2_score(znew, zpredridge))
covarbetaRid=sigma**2*np.linalg.inv(X.T.dot(X)+lambd*np.identity(X.shape[1])) #covariance matrix of the betas

#Uncomment the following line to print the variances of all the betas
#print(covarbetaRid[np.diag_indices(len(covarbetaRid))])

mseRidge=mean_squared_error(znew, zpredridge)
print("MSE (without resampling):",mseRidge)
    

####################### Ridge with Bootstrap######################

# Make data.
npoints = 100
xx = np.linspace(0, 1, npoints)
yy = np.linspace(0, 1, npoints)
xx, yy = np.meshgrid(xx,yy)
xx, yy = xx.reshape(-1, 1), yy.reshape(-1, 1)

#Parameters for Bootstrap

lambd=[0.00001, 0.0001,0.001, 0.1]
n_bootstraps=20
dim = (len(lambd), X.shape[1], n_bootstraps)

bootvarRid=np.zeros(dim)
bootmseRid=np.zeros((len(lambd),n_bootstraps))
bootr2Rid=np.zeros((len(lambd),n_bootstraps))

#Loop over different values of the regularization strength
for lambd_index, lam in enumerate(lambd):
  #Loop implementing Bootstrap  
  for i in range(n_bootstraps):
    #Training
    x_train, x_test, y_train, y_test = train_test_split(xx, yy, test_size=0.3)
    X_train=poly.fit_transform(np.c_[x_train, y_train])
    z_train = FrankeFunction(x_train, y_train)
    ztrnew = z_train + np.random.normal(0,sigma, z_train.shape)
    #Testing
    X_test=poly.fit_transform(np.c_[x_test, y_test])
    z_test = FrankeFunction(x_test, y_test)
    ztestnew = z_test + np.random.normal(0,sigma, z_test.shape)
    #Solution for Betas
    betaRidtest=np.linalg.pinv(X_train.T.dot(X_train)+lam*np.identity(X_train.shape[1])).dot(X_train.T).dot(ztrnew)
    zpredictest = X_test.dot(betaRidtest)
    
    #Validation scores
    covarbetRid=sigma**2*np.linalg.pinv(X_train.T.dot(X_train)+lam*np.identity(X_train.shape[1]))
    bootvarRid[lambd_index, :,i]=covarbetRid[np.diag_indices(len(covarbetRid))]
    bootmseRid[lambd_index, i]=mean_squared_error(ztestnew, zpredictest)
    bootr2Rid[lambd_index, i]=r2_score(ztestnew, zpredictest)

#Final validation scores averaged over Bootstrap resampling
betabootRid=np.mean(bootvarRid, axis=2)
msebootRid=np.mean(bootmseRid, axis=1)
r2bootRid=np.mean(bootr2Rid, axis=1)
 
#MSE and R2 for optimal regularization strength
print("MSE (with Bootstap):",np.min(msebootRid))
print("R2 score (with Bootstrap):",np.max(r2bootRid))


#Uncomment the following line to print the variances of all the betas (For all values of lambda)
#print(betabootRid)
  

####################### Ridge with cross-validation######################

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
   model = Ridge(alpha=lamb)
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
plt.title(r"$R^2$ score for Ridge {}th order".format(5))
plt.show()
    
#plots figure: MSE versus lambda
plt.semilogx(lambdas,np.absolute(CVmse), label="MSE")
plt.xlabel(r"$\lambda$", fontsize=18)
plt.ylabel(r"$MSE$", fontsize=18)
plt.title(r"$MSE$  for Ridge {}th order".format(5))
plt.show()
