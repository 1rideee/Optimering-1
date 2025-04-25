
from sklearn.metrics.pairwise import pairwise_kernels
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




def kernal_linear(x, y):
    '''
    Compute the linear kernel between two vectors x and y.
    '''

    return np.dot(x, y)

def kernal_gaussian(x, y, sigma=1):
    '''
    Compute the Gaussian kernel between two vectors x and y.
    '''

    return np.exp(-np.linalg.norm(x - y, 2)**2 / (2 * sigma**2))


def kernal_laplacian(x,y,sigma=1):
    '''
    Compute the Laplacian kernel between two vectors x and y.
    '''

    return np.exp(-np.linalg.norm(x - y,2) / sigma)
    

def kernal_inv_multiquadratic(x,y,sigma=1):
    '''
    Compute the inverse multiquadratic kernel between two vectors x and y.
    '''

    return 1 / np.sqrt(1 + np.linalg.norm(x - y, 2)**2 / sigma**2)



def f(alpha, A):
    return 0.5*np.dot(alpha, np.dot(A,alpha)) - np.sum(alpha)


def gradientf(alpha, A):
    return np.dot(A, alpha) - 1
    

def projection(alpha, y, C=1.0, tol=1e-6, max_iter=1000, delta=1e-3):  
    """
    Project the vector alpha onto the feasible region defined by the constraints.
    Parameters
    ----------
    alpha : numpy array
        The vector to be projected.
    y : numpy array
        The target vector.
    Y : numpy array
        The diagonal matrix of the target vector.
    C : float, optional
        The penalty parameter. The default is 1.
    tol : float, optional
        The tolerance for the convergence. The default is 1e-6.
    max_iter : int, optional
        The maximum number of iterations. The default is 100.
    delta : float, optional
        The step size for the binary search. The default is 1e-3.
    Returns
    -------
    projected_alpha : numpy array
        The projected vector.
    """

    beta = alpha.copy()
    low, high = -10, 10
    inner_low = np.dot(y, alpha_Lagrange(beta, low, y, C))
        
    if inner_low  > 0:
        cond= True
        while cond:
            high = low 
            low = low - delta
            cond = np.dot(y, alpha_Lagrange(beta, low, y, C)) < 0 and np.dot(y, alpha_Lagrange(beta, high, y, C))>0

    inner_high = np.dot(y, alpha_Lagrange(beta, high, y, C))
    if inner_high < 0:
        cond= True
        while cond:
            low = high
            high = high + delta
            cond = np.dot(y, alpha_Lagrange(beta, low, y, C)) < 0 and np.dot(y, alpha_Lagrange(beta, high, y, C))>0

    for _ in range(max_iter):

        nevner = (np.dot(y, alpha_Lagrange(beta, high, y, C))-np.dot(y, alpha_Lagrange(beta, low, y, C)))
        
        if nevner !=0:
            lambda_mid = high - (high-low)* np.dot(y, alpha_Lagrange(beta, high, y, C))/nevner
            
        else:
            lambda_mid = (low + high) / 2.0
        
        projected_alpha = alpha_Lagrange(beta, lambda_mid, y, C)
    
        constraint_value = np.dot(y, projected_alpha)
        
        # Check if the constraint is satisfied
        if abs(constraint_value) < tol:
            return projected_alpha  
    
        if constraint_value > 0:
            high = lambda_mid
        else:
            low = lambda_mid
            
    print("Warning: Maximum iterations reached without convergence. lambda:", lambda_mid)
    return alpha

def alpha_Lagrange(beta, lam, y, C):
    """
    Compute the Lagrange multiplier for the projection step.
    Parameters
    ----------
    beta : numpy array
        The current alpha vector.
        lam : float
        The Lagrange multiplier.
        y : numpy array
        The target vector.
        C : float, optional
        The penalty parameter. 
    Returns
    -------
    projected_alpha : numpy array
        The projected alpha vector.
    
    """
    
    return np.median(np.array([beta + lam*y, np.zeros(len(beta)), np.ones(len(beta))*C]), axis=0)

def gradient_descent(alpha0, G, y , tau0, niter, C=100, tol = 1e-7, gradientf=gradientf, projection=projection):
    '''
    Perform the gradient descent algorithm with a projected gradient step.

    Parameters
    ----------
    alpha0 : numpy array
        Initial guess for the alpha vector.
    G : numpy array
        The Gram matrix.
    y : numpy array
        The target vector.
    tau0 : float
        Initial step length.
    niter : int
        Number of iterations.
    C : float, optional
        The penalty parameter. The default is 100.
    tol : float, optional
        Tolerance for the convergence. The default is 1e-7.

    '''

    alpha = alpha0
    Y = np.diag(y)
    #Saves the A matrix to save on computation time
    A = np.dot(Y,np.dot(G,Y))
    tau = tau0

    for i in range(niter):
        d_k = projection(alpha - tau * gradientf(alpha, A), y=y, C=C) - alpha
        
        # Check for convergence when the largest component of the gradient is smaller than the tolerance
        if np.max(np.abs(d_k)) < tol:
            print("Converged after", i, "iterations")
            return alpha
        
        if i%500 == 0:
            print("Iteration", i, ":", np.max(np.abs(d_k))) 
        
        alpha = alpha + d_k 

        #Creates the Barzilai-Borwein step length
        tau = BB_step_length((alpha-d_k), alpha, gradientf, A, taumax=1e5, taumin=1e-5)
    
    print("Did not converge after", niter, "iterations", np.max(np.abs(d_k)))
    return alpha


def BB_step_length(ak, ak1, grad_f, A, taumax=1e5, taumin=1e-5):
    '''
    Determine the Barzilai-Borwein step length for the projected gradient descent
    algorithm.

    s^k = a ^{k+1} - a^k
    z^k = grad_f(a^{k+1}) - grad_f(a^k)
    '''
    
    
    nevner = np.dot((ak1 - ak), (grad_f(ak1, A) - grad_f(ak, A)))
    if  nevner<= 0:
        return taumax
    
    tau = np.dot((ak1 - ak), (ak1 - ak)) / nevner
    return max(min(tau, taumax), taumin)



def gradient_descent_linesearch(alpha0, G, y , tau0, niter, C=100, L = 10, tol = 1e-10, f=f, gradient=gradientf, project=projection):
    """"
    Gradient descent with backtracking line search
    
    Parameters
    ----------

    alpha0 : np.array
        Initial point
    G : np.array
        Kernel matrix
    y : np.array
        Labels
    tau0 : float
        Initial step size
    niter : int
        Number of iterations
    C : float
        Regularization parameter
    L : int
        Number of iterations before reference function is updated

    tol : float
        Tolerance for convergence

    Returns
    -------
    alpha : np.array
        Optimal alpha
    """
    alpha = alpha0
    Y = np.diag(y)
    A = np.dot(Y,np.dot(G,Y))
    tau = tau0

    f_ref = np.inf
    f_best = f(alpha, A)
    f_c = f_best
    ell = 0
    f_ks = np.zeros(niter)

    for i in range(niter):

        d_k = project(alpha - tau*gradient(alpha, A), y=y, C=C) - alpha

        if np.max(np.abs(d_k)) < tol:
            print("Converged after", i, "iterations")
            return alpha, f_ks
        
        if i%500 == 0:
            print("Iteration", i, ":", np.max(np.abs(d_k))) 
        
        f_k = f(alpha, A)
        f_ks[i] = f_k
        if f_k < f_best:
            f_best = f_k
            f_c = f_k
            ell = 0
        else:
            f_c = np.max([f_c, f_k])
            ell = ell + 1
        if ell == L:
            f_ref = f_c
            f_c = f_k
            ell = 0

        

        if f(alpha + d_k, A) > f_ref:
            dot1 = np.dot(d_k, np.dot(A, d_k))
            dot2 = np.dot(d_k, np.dot(A, alpha))
            dot3 = np.dot(alpha, np.dot(A, d_k))
            dot4 = np.sum(d_k)
            theta = - (0.5*dot2 + 0.5 *dot3 - dot4)/dot1
            # print("theta", theta, np.shape(alpha), np.shape(d_k))
            
        else:
            theta = 1

        alpha = alpha + theta * d_k
        
        tau = BB_step_length(alpha-theta*d_k, alpha, gradient, A, taumax=1e5, taumin=1e-5)


    print("Did not converge after", niter, "iterations")
    
    return alpha, f_ks




def plot_db(x,y, alpha, ker = kernal_linear, C=5):
    
    w = compute_w(alpha, y, x, kernel = ker)

    K = pairwise_kernels(x, metric=ker)  # Example kernel matrix
    b = compute_b(alpha, y, K, C)


    res = 100
    xx, yy = np.meshgrid(np.linspace(-8, 8, res), np.linspace(-8, 8, res))

    Z = np.array([w(np.array([xx.ravel()[i], yy.ravel()[i]])) for i in range(len(xx.ravel()))])+b

    Z = Z.reshape(res,res)
    print("Z", np.max(Z), np.min(Z))
    plt.contourf(xx, yy, Z, levels=[-100,0,100],  colors= ["blue","red"], alpha=0.5)
    

    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title("Data points with boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    #plotting a 3d surface fo Z
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Z')
    ax.set_title('3D Surface Plot of Z')
    plt.show()


def compute_w(alpha, y, X_train, kernel):
    """
    Compute w(x) for the nonlinear SVM.
    """

    def w_function(x):
        i = np.where((alpha > 0))[0]
        return np.sum(alpha[i] * y[i] * pairwise_kernels( [x],X_train[i], metric=kernel)[0])
    return w_function


def compute_b(alpha, y, K, C):
    """Estimate bias b using support vectors with 0 < alpha < C."""
    support_vector_indices = np.where((alpha > 0) & (alpha < C))[0]
    if len(support_vector_indices) == 0:
        print("Warning: No suitable support vectors found to compute b.")
        return 0.0

    b_vals = []
    i= support_vector_indices[0]

    return y[i] - np.sum(alpha * y * K[:, i])


def w_b(alpha, y, x, C):
    
    # I_s = np.where(alpha > 0 and alpha < C)
    I_s = [i for i in range(len(alpha)) if alpha[i] > 0 and alpha[i] < C]
    
    w = np.sum(alpha[I_s]*y[I_s]*x[I_s].T, axis=1) 
    b = y[I_s[0]] - np.dot(w, x[I_s[0]])
    return w, b
    


def plot_solution(x, y, w, b):

    plt.scatter(x[:,0], x[:,1], c=y)
    plt.plot([-3, 3], [(-b - w[0] * (-3)) / w[1], (-b - w[0] * 3) / w[1]], 'k-')
    #excact solution
    plt.plot([-3, 3], [(-1 - 1 * (-3)), (-1 - 1 * 3) ], 'r--')
    plt.show()