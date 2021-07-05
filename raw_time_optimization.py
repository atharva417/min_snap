from math import factorial as f
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import block_diag
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
def is_positive(t):
    for g in range(len(t)):
        if t[g]<0:
            return False
    return True
def form_Q(l,t): #l is index in time array
    Q_i=np.zeros(shape=(n,n))  #initlaizing
    for i in range(n):       
        for j in range(n):
            if (((i>3) and (j<=3))) or (((j>3) and (i<=3))):      #some of inital entries are zero as per article
                Q_i[i][j]=0
            elif (i<=3) and (j<=3):
                Q_i[i][j]=0
            else:
                r,c=i+1,j+1
                Q_i[i][j]= (f(r)*f(c)*(pow(t[l],r+c-7)-pow(t[l-1],r+c-7)))/(f(r-4)*f(c-4)*(r+c-7)) #formula from article
    #notice first integration is from t[0] to t[1], that is how i have used l
    return Q_i    
def comp_A(t):
    A=np.zeros(shape=((4*m)+2,n*m))

    for j in range(n*m):
            if j>=n:
                A[0][j],A[1][j],A[2][j]=0,0,0
            else:
                A[0][j],A[1][j],A[2][j]=pow(t[0],j),j*pow(t[0],j-1),j*(j-1)*pow(t[0],j-2)

    for j in range(n*m):
            if j<n*(m-1):
                A[3][j],A[4][j],A[5][j]=0,0,0
            else:
                h=n*(m-1)
                A[3][j],A[4][j],A[5][j]=pow(t[3],j-h),(j-h)*pow(t[3],j-h-1),(j-h)*(j-h-1)*pow(t[3],j-h-2)
    z=[]
    for i in range(1,m):
        h=[]
        for j in range(n*m):
            if (j<((i-1)*n)) or (j>=(i*n)):
                h.append(0)
            else:
                h.append(pow(t[i],j-((i-1)*n)))
        z.append(h)    
    A[6:(6+m-1)]=z
    pva_const=[]
    for i in range(1,m):
        x_i,v_i,a_i=[],[],[]
        for j in range(n*m):
            if (j<((i-1)*n)) or (j>=((i+1)*n)):
                x_i.append(0)
                v_i.append(0)
                a_i.append(0)
                
            elif (j<((i)*n)) and (j>=((i-1)*n)):
                x_i.append(pow(t[i],j-((i-1)*n)))
                v_i.append((j-((i-1)*n))*pow(t[i],j-1-((i-1)*n)))
                a_i.append((j-1-((i-1)*n))*(j-((i-1)*n))*pow(t[i],j-2-((i-1)*n)))
            else:
                x_i.append((-1)*pow(t[i],j-((i)*n)))
                v_i.append((-1)*(j-((i)*n))*pow(t[i],j-1-((i)*n)))
                a_i.append((-1)*(j-1-((i)*n))*(j-((i)*n))*pow(t[i],j-2-((i)*n)))
        pva_i=[x_i,v_i,a_i]        
        pva_const=pva_const+pva_i
    A[(6+m-1):]=pva_const
    return A    
   

def cost_function(t_input):
    global p_x_final
    global p_y_final
    global p_z_final
    t=[0.2]
    sum=0.2
    for i in range(len(t_input)):
        sum+=t_input[i]
        t.append(sum)
    #print(t)
    Q_1=form_Q(1,t)   
    Q_2=form_Q(2,t)
    Q_3=form_Q(3,t)  

    Q=block_diag(Q_1,Q_2,Q_3) #converts to block diagonal form in given order
    Q=Q+(0.0001*np.identity(n*m))

    A=comp_A(t)

    b_x=[x[0],0,0,x[m],0,0,x[1],x[2]]
    b_x.extend(np.zeros(shape=(3*(m-1))))
    b_y=[y[0],0,0,y[m],0,0,y[1],y[2]]
    b_y.extend(np.zeros(shape=(3*(m-1))))
    b_z=[z[0],0,0,z[m],0,0,z[1],z[2]]
    b_z.extend(np.zeros(shape=(3*(m-1))))
    q=np.zeros(shape=(n*m,1)).reshape((n*m,))
    G=np.zeros(shape=((4*m)+2,n*m))
    h=np.zeros(shape=((4*m)+2,1)).reshape(((4*m)+2,))

    p_x=solve_qp(Q, q,G,h, A, b_x)
    p_y=solve_qp(Q, q,G,h, A, b_y)
    p_z=solve_qp(Q, q,G,h, A, b_z)
    p_x_final=np.copy(p_x)
    p_y_final=np.copy(p_y)
    p_z_final=np.copy(p_z)
    K=10000000
    J_x=(0.00001*(np.matmul(np.matmul(np.transpose(p_x),Q),p_x))) +(0.00001*(np.matmul(np.matmul(np.transpose(p_y),Q),p_y)))+(0.00001*(np.matmul(np.matmul(np.transpose(p_z),Q),p_z)))+ (K*(t[-1]-t[0]))
    return J_x/100000

def grad_func(t):
    gradient=np.zeros(np.array(t).shape)
    h=0.0001
    t_comp=np.copy(t)
    for s in range(len(t)):
        t_comp[s]=t_comp[s]+h
        a=cost_function(t_comp)
        b=cost_function(t)
        gradient[s]=((a-b)/h)
        
        t[s]=t[s]-h
    return gradient

def gradient_descent(max_iterations,threshold,t_init,learning_rate=0.05):
    
    w = np.array(t_init)
    w_history = np.copy(w)
    f_history = cost_function(w)
    print('inital cost=',f_history)
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    
    while  (i<max_iterations) and (diff>threshold) :
        
        w_history = np.copy(w)
        grad=grad_func(w)
        delta_w=-learning_rate*grad
        w = w+delta_w       
        i+=1
        cost=cost_function(w)
        diff = abs(cost-cost_function(w_history))
      
    print('\nNo: of iterations : ',i)
    
    print('\nAfter Optimization of total time and Snap :')
    print('Cost after gradient descent :',cost)
    return w
def give_intervals(t):
    m=[]
    for j in range(len(t)-1):
        m.append(t[j+1]-t[j])
    return m


n=8 #considering degree is 7, there will be 1+(degree) coeffs so i used 8 
m=3 #considering there are m segments


t_initial=[0.2,0.8,1,1.2] 
print('Initial total time :',t_initial[-1]-t_initial[0])

x = [2,0,-2,0]
y = [0,2,0,-2]
z = [0,1,2,0]
p_x_final=np.empty(shape=(n*m,1))
p_y_final=np.empty(shape=(n*m,1))
p_z_final=np.empty(shape=(n*m,1))
t_interval=gradient_descent(500,0.005,give_intervals(t_initial),learning_rate=0.00001)

#getting time segment interval from gradient descent soln
t=[0.2]
sum=0.2
for i in range(len(t_interval)):
    sum+=t_interval[i]
    t.append(sum)
print('optimal time segmentation after gradient descent :',t)
print('total time :',(t[-1]-t[0]))

#PLOTTING

plt.figure(figsize=(10,5))
ax = plt.axes(projection ='3d')

ax.scatter(x, y, z, 'b',marker='o')
for v in range(m):
    w,u,a=[],[],[]
    
    r=np.linspace(t[v],t[v+1],100)
    for i in range(100):
        g,e,q=0,0,0
        for j in range(n*v,(v+1)*n):
            g=g+(p_x_final[j]*pow(r[i],j-(n*v)))
            e=e+(p_y_final[j]*pow(r[i],j-(n*v)))
            q=q+(p_z_final[j]*pow(r[i],j-(n*v)))
        w.append(g)
        u.append(e)
        a.append(q)
    ax.plot3D(w, u, a, 'r')

plt.show()
