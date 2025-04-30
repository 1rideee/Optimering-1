
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.optimize import approx_fprime


def TestLinear(w,b,n_A,n_B,margin,**kwargs):
    '''
    Sample code for testing 
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
    

def kernal_inv_multiquadratic(x,y,sigma=1, s=1):
    '''
    Compute the inverse multiquadratic kernel between two vectors x and y.
    '''

    return 1 /(sigma**2 + np.linalg.norm(x - y, 2)**2 )**s



def f(alpha, A):
    """
    Compute the objective function value for the dual problem.
    Parameters
    ----------
    alpha : numpy array
        The dual variables.
    A : numpy array
        The Gram matrix.
    Returns
    -------
    f_value : float
        The value of the objective function.
    """
    return 0.5*np.dot(alpha, np.dot(A,alpha)) - np.sum(alpha)


def gradientf(alpha, A):
    """
    Compute the gradient of the objective function.
    Parameters
    ----------
    alpha : numpy array
        The dual variables.
    A : numpy array
        The Gram matrix.
    Returns
    -------
    grad : numpy array
        The gradient of the objective function.
    """
    # Analytic gradient of the objective function
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
            
    #Optional print for warning when projection does not converge
    # print("Warning: Maximum iterations reached without convergence. lambda:", lambda_mid)
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
        
        # if i%500 == 0:
        #     print("Iteration", i, ":", np.max(np.abs(d_k))) 
        
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
            
        else:
            theta = 1

        alpha = alpha + theta * d_k
        
        tau = BB_step_length(alpha-theta*d_k, alpha, gradient, A, taumax=1e5, taumin=1e-5)


    print("Did not converge after", niter, "iterations")
    
    return alpha, f_ks



def plot_db(x,y, alpha, ker = kernal_linear, C=5):
    """
    Plots the decision boundary of the SVM.

    Parameters
    ----------
    x : numpy array
        The input data.
    y : numpy array
        The labels.
    alpha : numpy array
        The dual variables.
    ker : function, optional
        The kernel function. The default is kernal_linear.
    C : float, optional
        The penalty parameter. 

    Returns
    -------
    None.
    """
    
    w = compute_w(alpha, y, x, kernel = ker)

    K = pairwise_kernels(x, metric=ker)  # Example kernel matrix
    b = compute_b(alpha, y, K, C)


    res = 100
    
    xx, yy = np.meshgrid(np.linspace(np.min(x)-1, np.max(x)+1, res), np.linspace(np.min(x)-1, np.max(x)+1, res))

    Z = np.array([w(np.array([xx.ravel()[i], yy.ravel()[i]])) for i in range(len(xx.ravel()))])+b

    Z = Z.reshape(res,res)
    # print("Z", np.max(Z), np.min(Z))
    plt.contourf(xx, yy, Z, levels=[-100,0,100],  colors= ["blue","red"], alpha=0.5)
    

    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    # plt.title("Data points with boundary")
    # plt.xlabel("Feature 1")
    # plt.ylabel("Feature 2")
    plt.show()    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx, yy, Z, cmap='viridis', edgecolor='none')
    # ax.set_xlabel('X1')
    # ax.set_ylabel('X2')
    # ax.set_zlabel('Z')
    # ax.set_title('3D Surface Plot of Z')
    # plt.show()


def compute_w(alpha, y, X_train, kernel):
    """
    Compute w(x) for the nonlinear SVM.

    Parameters
    ----------

    alpha : numpy array
        The dual variables.
    y : numpy array
        The labels.
    X_train : numpy array
        The training data.
    kernel : function
        The kernel function.

    Returns
    -------
    w_function : function
        A function that computes w(x) for a given x.
    
    """

    def w_function(x):
        i = np.where((alpha > 0))[0]
        return np.sum(alpha[i] * y[i] * pairwise_kernels( [x],X_train[i], metric=kernel)[0])
    return w_function


def compute_b(alpha, y, K, C):
    """Estimate bias b using support vectors with 0 < alpha < C.
    
    Parameters
    ----------
    alpha : numpy array
        The dual variables.
    y : numpy array
        The labels.
    K : numpy array
        The kernel matrix.
    C : float
        The penalty parameter.

    Returns
    -------
    b : float
        The bias term.
    """
    svm = np.where((alpha > 0) & (alpha < C))[0]
    if len(svm) == 0:
        print("Warning: No suitable support vectors found to compute b.")
        return 0.0

    i = svm[0]

    return y[i] - np.sum(alpha * y * K[:, i])



def w_b(alpha, y, x, C):
    """Compute w and b for the linear SVM using the dual solution alpha.
    
    Parameters
    ----------
    alpha : numpy array
        The dual variables.
    y : numpy array
        The labels.
        
    x : numpy array
        The input data.
    C : float
        The penalty parameter.
    Returns
    -------
    w : numpy array
        The weight vector.
    b : float
        The bias term.
    """

    I_s = [i for i in range(len(alpha)) if alpha[i] > 0 and alpha[i] < C]
    
    w = np.sum(alpha[I_s]*y[I_s]*x[I_s].T, axis=1) 
    b = y[I_s[0]] - np.dot(w, x[I_s[0]])
    return w, b




def plot_solution(x, y, w, b):
    """
    Plot the decision boundary and the data points.

    ----------
    Returns:
        None
    """

    plt.scatter(x[:,0], x[:,1], c=y)
    plt.plot([-3, 3], [(-b - w[0] * (-3)) / w[1], (-b - w[0] * 3) / w[1]], 'k-', label='Decision boundary')
    #excact solution
    plt.plot([-3, 3], [(-1 - 1 * (-3)), (-1 - 1 * 3) ], 'r--', label='Exact solution')
    plt.legend()
    plt.show()

def test_kernel(alpha0, x, y, ker=kernal_gaussian, niter=1000, C=1, tau0=0.1, tol=1e-7, plot=True):
    """
    Test function to test the kernelized gradient descent.
    This function performs gradient descent on the kernelized SVM problem using the specified kernel function.
    It initializes the alpha values, computes the kernel matrix, and performs gradient descent, 
    and plots the result if needed.

    Parameters:
    alpha0 : np.array
        Initial alpha values.
    x : np.array
        Input data points.
    y : np.array
        Labels for the data points.
    ker : function
        Kernel function to be used.
    niter : int
        Number of iterations for gradient descent.
    C : float
        Regularization parameter.
    tau0 : float
        Initial step size for gradient descent.
    tol : float
        Tolerance for convergence.
    plot : bool
        Whether to plot the decision boundary or not.
    """
    G = pairwise_kernels(x, metric = ker)  
    alpha, fk = gradient_descent_linesearch(alpha0, G, y, tau0=tau0, niter=niter, C=C, tol=tol)
    
    if plot:
        plot_db(x, y, alpha, ker = ker, C=C)


def projection_AL(vector, proj_par):
    lower_bounds = proj_par[0]
    upper_bounds = proj_par[1]
    dimension = len(vector)
    proj_vector = np.array([])
    
    for k in range(0, dimension):
        if vector[k]<lower_bounds[k]:
            proj_vector = np.append(proj_vector, np.array([lower_bounds[k]]))
        elif vector[k] > upper_bounds[k]:
            proj_vector = np.append(proj_vector, np.array([upper_bounds[k]]))
        else:
            proj_vector = np.append(proj_vector, np.array([vector[k]]))
    return proj_vector

def general_BB_steplength(vec_k, vec_k1, grad, grad_par, taumax=1e5, taumin=1e-5):
    nevner = np.dot((vec_k1 - vec_k), (grad(vec_k1, grad_par) - grad(vec_k, grad_par)))
    if  nevner<= 0:
        return taumax
    tau = np.dot((vec_k1 - vec_k), (vec_k1 - vec_k)) / nevner
    return max(min(tau, taumax), taumin)

def constraints(vec, constr_par = False):
    if constr_par:
        x, y  = constr_par[0], constr_par[1]
    d = 2
    M = len(x)

    w = vec[0:d]
    b = vec[d]
    xi = vec[d+1:d+M+1]
    s = vec[d+M+1:]

    c_vec = np.array([])
    for i in range(0, len(x)):
        c_vec = np.append(c_vec, np.array([y[i]*(np.inner(w, x[i]) + b) + xi[i] -s[i]-1]))
    
    return c_vec

def BCLM(vec_0, lambd_0, mu_0, tol_1, tol_2, maxiter, func, func_par, constr, constr_par, grad, grad_par, project, project_par, linesearch, linesearch_par): #Algoritme 17.4 i boka
    
    tol_1_k = 1/mu_0
    tol_2_k = 1/mu_0**(0.1)

    vec_k = vec_0
    lambd_k = lambd_0
    mu_k = mu_0

    grad_par[0] = lambd_k
    grad_par[1] = mu_k
    func_par[0] = lambd_k
    func_par[1] = mu_k

    for _ in range(0, maxiter):
        print("iterasjon BCLM", _)
        tau_0 = 1
        projected_gradient_method = general_projected_gradient_linesearch(vec_k, tau_0, func, func_par, grad, grad_par, project, project_par, linesearch, linesearch_par, tol = tol_1_k, L = 10)

        #general_projected_gradient_linesearch(vec_0, tau_0, func, func_par, grad, grad_par, project, project_par, linesearch, linesearch_par, tol, L = 10)

        vec_k= projected_gradient_method[0]
        d_k = projected_gradient_method[2]

        c_k = constr(vec_k, constr_par)
        c_k_norm = np.linalg.norm(c_k)
        print(vec_k,"vecc BCL;")
        print(c_k_norm,"ck")

        if c_k_norm <= tol_2_k:
            print("ck mindre")
            if c_k_norm <= tol_2 and np.linalg.norm(d_k) <= tol_1:
                return vec_k, lambd_k
            
            lambd_k = lambd_k - mu_k*c_k
            tol_1_k = tol_1_k/mu_k
            tol_2_k = tol_2_k/mu_k**(0.9)
            print(tol_1_k,tol_2_k,"tols")

            grad_par[0] = lambd_k
            func_par[0] = lambd_k
            linesearch_par[0] = lambd_k

        
        else:
            print("else")
            mu_k = 100*mu_k
            tol_1_k = tol_1_k/mu_k
            tol_2_k = tol_2_k/mu_k**(0.1)
            print(tol_1_k,tol_2_k,"tols")
            grad_par[1] = mu_k
            func_par[1] = mu_k
            linesearch_par[1] = mu_k

    return "Ingen konvergens"



def AL(vec, AL_par): #kontroller at dette er rett

    lambd = AL_par[0]
    mu = AL_par[1]

    d = AL_par[2]
    M = AL_par[3]

    x = AL_par[4]
    y = AL_par[5]

    C = AL_par[6]
    # print(vec,"al")
    w = vec[0:d]
    b = vec[d]
    xi = vec[d+1:d+1+M]
    s = vec[d+1+M:]
    
    AL = 0.5*np.linalg.norm(w)**2
    for i in range(0, len(xi)):
        indreprod = np.inner(w, x[i])
        AL = AL + C*xi[i] - lambd[i]*(y[i]*(indreprod + b) + xi[i] - s[i] - 1) + 0.5*mu*(y[i]*(indreprod + b) + xi[i] - s[i] - 1)**2
    return AL




def grad_AL(vec, gradAL_par): #Kontroller at dette er rett
    
    lambd = gradAL_par[0]
    mu = gradAL_par[1]

    d = gradAL_par[2]
    M = gradAL_par[3]

    x = gradAL_par[4]
    y = gradAL_par[5]

    C = gradAL_par[6]

    w = vec[0:d]
    b = vec[d]
    xi = vec[d+1:d+1+M]
    s = vec[d+1+M:]
    
    grad_AL = np.array([])
    
    #Elements from w
    for k in range(0, d):
        grad_k = w[k]
        for i in range(0, M):
            indresum = 0
            for l in range(0, d):
                indresum = indresum + 2*x[i][k]*x[i][l]*w[l]
            grad_k = grad_k - lambd[i]*y[i]*x[i][k] + 0.5*mu*(y[i]**2 * indresum + 2*y[i]**2 * b * x[i][k] + 2*y[i]*xi[i]*x[i][k] - 2*y[i]*s[i]*x[i][k] -2*y[i]*x[i][k])
        grad_AL = np.append(grad_AL, np.array([grad_k]))
    
    #Elements from b
    grad_b = 0
    for i in range(0, M):
        grad_b = grad_b - lambd[i]*y[i] + 0.5*mu*(2*y[i]**2*b + 2*y[i]**2*np.inner(w, x[i]) + 2*y[i]*xi[i] - 2*y[i]*s[i] - 2*y[i])
    grad_AL = np.append(grad_AL, np.array([grad_b]))

    #Elements from xi
    for i in range(0, M):
        grad_xi = C - lambd[i] + 0.5*mu*(2*xi[i] + 2*y[i]*np.inner(w, x[i]) + 2*y[i]*b - 2*s[i] - 2)
        grad_AL = np.append(grad_AL, np.array([grad_xi]))

    #Elements from s
    for i in range(0, M):
        grad_s = -lambd[i] + 0.5*mu*(2*s[i] - 2*y[i]*np.inner(w, x[i]) - 2*y[i]*b - 2*xi[i] + 2)
        grad_AL = np.append(grad_AL, np.array([grad_s]))
    
    return grad_AL



def general_projected_gradient_linesearch(vec_0, tau_0, func, func_par, grad, grad_par, project, project_par, linesearch, linesearch_par, tol, L = 10, niter = 1000):
    """
    General projected gradient descent with backtracking line search.

    Parameters
    ----------
    vec_0 : numpy array
        Initial point.
    tau_0 : float
        Initial step size.
    func : function
        Objective function to be minimized.
    func_par : list
        Parameters for the objective function.
    grad : function
        Gradient of the objective function.
    grad_par : list
        Parameters for the gradient function.
    project : function
        Projection function.
    project_par : list
        Parameters for the projection function.
    linesearch : function
        Line search function.
    linesearch_par : list
        Parameters for the line search function.
    tol : float
        Tolerance for convergence.
    L : int, optional
        Number of iterations before reference function is updated.
        The default is 10.
    niter : int, optional
        Number of iterations.
        The default is 1000.
    Returns
    -------
    vec : numpy array
        Optimal point.
    f_ks : numpy array
        Function values at each iteration.
    d_k : numpy array
        Search direction at the last iteration.
    """
    vec = vec_0
    tau = tau_0
    
    f_ref = np.inf
    f_best = func(vec, func_par)
    f_c = f_best
    ell = 0
    f_ks = np.zeros(niter)

    for i in range(niter):
        
        # if i == 0:
            # print(vec,tau,grad_par,project_par)
        d_k = project(vec -  tau*grad(vec, grad_par), project_par) - vec
        
        if (np.linalg.norm(d_k)) < tol:
            print("Converged after", i, "iterations")
            # print(vec)
            return vec, f_ks, d_k
        
        if i%500 == 0:
            print("Iteration", i, ":", np.max(np.abs(d_k))) 
        
        f_k = func(vec, func_par)
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
        # print(ell,"l")
        if func(vec + d_k, func_par) > f_ref:
            theta = linesearch(vec, d_k, linesearch_par)
            
        else:
            theta = 1
            
        vec_temp=vec
        vec = vec + theta * d_k
        # print(np.linalg.norm(vec_temp-vec,1),"vec-vek")
        tau = general_BB_steplength(vec-theta*d_k, vec, grad, grad_par, taumax=1e5, taumin=1e-5)



    print("Did not converge after", niter, "iterations")
    
    return vec, f_ks, d_k


def linesearch_AL(vec, d_k, linesearch_par):
    """
    Perform a line search to find the optimal step size for the projected gradient descent.

    Parameters
    ----------
    vec : numpy array
        The current point.
    d_k : numpy array
        The search direction.
    linesearch_par : list
        Parameters for the line search.
        [lambda, mu, d, M, x, y, C]
    Returns
    -------
    theta : float
        The optimal step size.
    """
    
    lambd = linesearch_par[0]
    mu = linesearch_par[1]

    d = linesearch_par[2]
    M = linesearch_par[3]

    x = linesearch_par[4]
    y = linesearch_par[5]

    C = linesearch_par[6]
    # print(vec,"line")
    w = vec[0:d]
    b = vec[d]
    xi = vec[d+1:d+1+M]
    s = vec[d+1+M:]

    d_w = d_k[0:d]
    d_b = d_k[d]
    d_xi = d_k[d+1:d+1+M]
    d_s = d_k[d+1+M:]
    
    A = np.linalg.norm(d_k[0:d],2)**2
    B = np.inner(w, d_w) + C*np.sum(d_xi)

    for i in range(0, M):
        indprod_1 = np.inner(d_w, x[i])
        indprod_2 = np.inner(w, x[i])
        
        A = A + mu* (indprod_1**2 + d_k[d]**2 + d_k[d+i]**2 + d_k[d+M+i]**2 + 2*d_k[d]*indprod_1 + 2*y[i]*d_k[d+i]*indprod_1 - 2*y[i]*d_k[d+M+i]*indprod_1 + 2*y[i]*d_k[d]*d_k[d+i] - 2*y[i]*d_k[d]*d_k[d+M+i] - 2*d_k[d+i]*d_k[d+M+i])

        B = B - lambd[i]* (y[i]*(indprod_1 + d_k[d]) + d_k[d+i] - d_k[d+M+i]) + mu*(indprod_2 * indprod_1 + b*d_k[d] + xi[i]*d_k[d+i] + s[i]*d_k[d+M+i] + b*indprod_1 + d_k[d]*indprod_2 + y[i]*xi[i]*indprod_1 + y[i]*d_k[d+i]*indprod_2 - y[i]*s[i]*indprod_1 - y[i]*d_k[d+M+i]*indprod_2 - y[i]*indprod_1 + y[i]*d_k[d]*xi[i] + y[i]*d_k[d+i]*b - y[i]*d_k[d+M+i]*b - y[i]*d_k[d]*s[i] - y[i]*d_k[d] - d_k[d+i]*s[i] - d_k[d+M+i]*xi[i])
    
    theta = -B/A
    
    return theta
