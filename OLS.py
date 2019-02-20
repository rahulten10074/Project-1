from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
fig = plt.figure()
ax = fig.gca(projection='3d')


############ OLS without resampling##################


# Make data
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2) 
    return term1 + term2 + term3 + term4 

#creating noise term
sigma = 1
z = FrankeFunction(x, y)
z = z + 0.01*np.random.normal(0,sigma,(20,20)) # z with noise


#Creating design matrix
poly=PolynomialFeatures(degree=5) 
X=poly.fit_transform(np.c_[x.ravel(), y.ravel()])

#Solution for parameters, beta
beta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z.ravel())



# Calculating MSE,R2 and Variances of Beta
zpredict = X.dot(beta).reshape(20,20)
print("Mean squared error (no resampling):", mean_squared_error(z, zpredict))
print("R2 score (no resampling):", r2_score(z, zpredict))         


covarbeta=sigma**2*np.linalg.inv(X.T.dot(X)) #covariance matrix of the betas

#Uncomment the following line to print the variances of all the betas

#print("variance of betas (no resampling):",covarbeta[np.diag_indices(len(covarbeta))])


############# plotting the fit###############################

surf = ax.plot_surface(x, y, zpredict, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('X')
ax.set_ylabel('Y')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()   

################# K-fold cross validation#####################

kf = KFold(n_splits=5,shuffle=True)

#Making data
xx = np.arange(0, 1, 0.0005)
yy = np.arange(0, 1, 0.0005)
kf.get_n_splits(xx)


kfoldmse=[]
kfoldr2=[]
kfoldvar=[]

#Loop implementing cross-validation
for i,j in kf.split(xx):
    x_train= xx[i]
    y_train= yy[i]
    x_test= xx[j]
    y_test= yy[j]
    
    #Training
    X_train=poly.fit_transform(np.c_[x_train, y_train])
    z_train = FrankeFunction(x_train, y_train)
    ztrnew = z_train + 0.01*np.random.normal(0,sigma,(len(z_train),))  #Adding noise
    betatrain=np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(ztrnew)

    #Testing
    X_test=poly.fit_transform(np.c_[x_test, y_test])
    z_pred = X_test.dot(betatrain)
    ztest = FrankeFunction(x_test, y_test)
    
    #Validation scores
    kfoldmse.append(mean_squared_error(ztest, z_pred))
    kfoldr2.append(r2_score(ztest, z_pred))
    covarbetrain=sigma**2*np.linalg.pinv(X_train.T.dot(X_train))
    kfoldvar.append(covarbetrain[np.diag_indices(len(covarbetrain))])


kfoldvar=np.mean(kfoldvar, axis=0)
mse=np.mean(kfoldmse)
r2=np.mean(kfoldr2)

print ("Mean squared error (cross-validation):",mse)    
print ("R2 score (cross-validation):",r2)

#Uncomment the following line to print the variances of all the betas
#print ("variance of betas (cross-validation):",kfoldvar)



################################   Bootstrap #####################################



# Make data.
npoints = 100
xx = np.linspace(0, 1, npoints)
yy = np.linspace(0, 1, npoints)
xx, yy = np.meshgrid(xx,yy)
xx, yy = xx.reshape(-1, 1), yy.reshape(-1, 1)

n_bootstraps=20

dim = (X.shape[1], n_bootstraps)
bootvar=np.zeros(dim)
bootmse=np.zeros(n_bootstraps)
bootr2=np.zeros(n_bootstraps)


for i in range(n_bootstraps):
    
    #initialisation
    x_train, x_test, y_train, y_test = train_test_split(xx, yy, test_size=0.3)
    #Trainning
    X_train=poly.fit_transform(np.c_[x_train, y_train])
    z_train = FrankeFunction(x_train, y_train)
    ztrnew = z_train + np.random.normal(0,sigma, z_train.shape)
    
    #Testing
    X_test=poly.fit_transform(np.c_[x_test, y_test])
    z_test = FrankeFunction(x_test, y_test)
    ztestnew = z_test + np.random.normal(0,sigma, z_test.shape)
    
    #beta parameters 
    betatrain=np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(ztrnew) #pinv: pseudo inversion
    zpredictest = X_test.dot(betatrain)# to make predictns on the test/unknown data
    
    #Covariance, MSE, R2_SCORE  
    covarbetrain=sigma**2*np.linalg.pinv(X_train.T.dot(X_train))
    bootvar[:,i]=covarbetrain[np.diag_indices(len(covarbetrain))]
    bootmse[i]=mean_squared_error(ztestnew, zpredictest)
    bootr2[i]=r2_score(ztestnew, zpredictest)
    
betaboot=np.mean(bootvar, axis=1)
mseboot=np.mean(bootmse)
r2boot=np.mean(bootr2)



print("Mean squared error (Bootstrap):",mseboot)
print("R2 score (Bootstrap):",r2boot)


#Uncomment the following line to print the variances of all the betas
#print("variance of betas (bootstrap):",betaboot)
