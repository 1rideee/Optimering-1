#Functions file for the main program



import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng


def TestLinear(w,b,n_A,n_B,margin,**kwargs):
    '''
    

    Parameters
    ----------
    w : non-zero vector
        normal vector defining a hyperplane
    b : real number
        offset of the hyperplane
    n_A : integer
        number of additional samples from class A
    n_B: integer
        number of additional samples from class B
    margin : positive real
        desired margin for the samples
        
    Optional Parameters
    -------------------
    seed : integer
        seed for the random number generator
        default value : 18
    sigma : positive real
        standard deviation for the normal distribution
        default value : 1.
    shape : positive real
        shape parameter for the Gamma distribution
        default value : 1.
    scale : positive real
        scale parameter for the Gamma distribution
        default value : 1.

    Returns
    -------
    list_A, list_B : lists of vectors
        list_A contains n_A vectors all lying on one side of the hyperplane H(w,-b).
        The distance to the hyperplane is margin + a sample of a Gamma distribution.
        In the plane normal to w, the points follow a normal distribution.
        One of the vectors acts as a support vector with precise margin gamma.
        list_B contains n_B vectors, produced in a similar way, lying on the
        opposite side of the hyperplane.

    '''
    
    # read out additional keyword arguments
    seed = kwargs.get("seed",18)
    shape = kwargs.get("shape",1.)
    scale = kwargs.get("scale",1.)
    sigma = kwargs.get("sigma",1.)
    
    # read out the number of dimensions
    d = w.size
    
    # rescale w to length 1
    norm_w = np.linalg.norm(w)
    w = w/norm_w
    b = b/norm_w

    # initialise a random number generator
    rng = default_rng(seed)
    
    # initialise an empty list
    list_A = []    
    # draw samples for class A
    for _ in range(n_A):
        # draw n_A samples of a d-dimensional normal distribution
        vec = rng.normal(size=d,scale=sigma)
        # draw n_A samples of a Gamma distribution
        dist = rng.gamma(shape,scale)
        # project vec onto w^\perp
        vec += -np.inner(vec,w)*w
        # add (dist+margin+b)*w to vec
        vec += (dist+margin-b)*w
        # append the vector vec to list_A
        list_A.append(vec)
        
    # initialise an empty list
    list_B = []    
    # draw samples for class A
    for _ in range(n_B):
        # draw n_B samples of a d-dimensional normal distribution
        vec = rng.normal(size=d,scale=sigma)
        # draw n_A samples of a Gamma distribution
        dist = rng.gamma(shape,scale)
        # project vec onto w^\perp
        vec += -np.inner(vec,w)*w
        # add -(dist+margin-b)*w to vec
        vec += (-b-dist-margin)*w
        # append the vector vec to list_B
        list_B.append(vec)
    
    # choose a random vector of each list and force it to be a support vector
    vec = rng.normal(size=d,scale=sigma)
    vec += -np.inner(vec,w)*w
    supp_A = rng.integers(0,n_A)
    list_A[supp_A] = vec+(margin-b)*w
    supp_B = rng.integers(0,n_B)
    list_B[supp_B] = vec+(-b-margin)*w

    return(list_A,list_B)


if __name__ == "__main__":
    w = np.array([1.,1.])
    b = 1.
    n_A = 10
    n_B = 8
    margin = 5.e-1
    listA,listB = TestLinear(w,b,n_A,n_B,margin)
    [plt.scatter(x[0],x[1],color="r") for x in listA]
    [plt.scatter(x[0],x[1],color="b") for x in listB]
    plt.show()



def projected_grad_desc():

    pass


def BB_step_length(ak, ak1, grad_f):
    '''
    Determine the Barzilai-Borwein step length for the projected gradient descent
    algorithm.

    s^k = a ^{k+1} - a^k
    z^k = grad_f(a^{k+1}) - grad_f(a^k)
    '''

    if np.dot(ak1 - ak, grad_f(ak1) - grad_f(ak)) <= 0:
        return 1
    
    return np.dot(ak1 - ak, ak1 - ak) / np.dot(ak1 - ak, grad_f(ak1) - grad_f(ak))
    

def kernal_gaussian(x, y, sigma):
    '''
    Compute the Gaussian kernel between two vectors x and y.
    '''

    return np.exp(-np.linalg.norm(x - y, 2)**2 / (2 * sigma**2))


def kernal_laplacian(x,y,sigma):
    '''
    Compute the Laplacian kernel between two vectors x and y.
    '''

    return np.exp(-np.linalg.norm(x - y,2) / sigma)
    

def kernal_inv_multiquadratic(x,y,sigma):
    '''
    Compute the inverse multiquadratic kernel between two vectors x and y.
    '''

    return 1 / np.sqrt(1 + np.linalg.norm(x - y, 2)**2 / sigma**2)



def f(alpha, A):
    return 0.5*np.dot(alpha, np.dot(A,alpha)) - np.sum(alpha)


def gradientf(alpha, A):
    return np.dot(A, alpha) - 1
    

def gradient_descent(alpha0, G, y , tau, niter):
    alpha = alpha0
    Y = np.diag(y)
    A = np.dot(Y,np.dot(G,Y))


    for i in range(niter):
        d_k = projection(alpha - tau*gradientf(alpha, A), y=y, Y=Y) - alpha
        alpha = alpha + d_k 
        # tau * gradientf(alpha, G, Y)

    return alpha


def projection(alpha, y, Y, C=1.0, tol=1e-6, max_iter=100, delta=1e-3):  
    beta = alpha.copy()
    
    
    low, high = -10, 10  
    for _ in range(max_iter):
        


        inner_low = np.dot(y, alpha_Lagrange(beta, low, Y, C))
        inner_high = np.dot(y, alpha_Lagrange(beta, high, Y, C))

        if inner_low  > 0:
            while np.dot(y, alpha_Lagrange(beta, low, Y, C)) < 0 and np.dot(y, alpha_Lagrange(beta, high, Y, C))>0:
                high = low 
                low = low - delta

        if inner_high < 0:
            while np.dot(y, alpha_Lagrange(beta, low, Y, C)) < 0 and np.dot(y, alpha_Lagrange(beta, high, Y, C))>0:
                low = high
                high = high + delta

        lambda_mid = (low + high) / 2.0
        
        projected_alpha = alpha_Lagrange(beta, lambda_mid, Y, C)
    
        constraint_value = np.dot(y, projected_alpha)

        if abs(constraint_value) < tol:
            return projected_alpha  
    
        if constraint_value > 0:
            high = lambda_mid
        else:
            low = lambda_mid
            
    return projected_alpha, "Never converged" 


def alpha_Lagrange(beta, lam, Y, C=1):
    projected_alpha = np.zeros(len(beta))
    for i in range(len(beta)):
            projected_alpha[i] = np.median([beta[i] + lam * Y[i][i], 0, C])
    return projected_alpha