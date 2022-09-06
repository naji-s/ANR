import numpy as np
def haar_measure(n):
    """A Random matrix distributed with Haar measure"""
    z = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.sqrt(2.0)
    q,r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d/np.absolute(d)
    q = np.multiply(q,ph,q)
    return q
global flag,A,B
flag=False

def trace_check(n):
    global flag,A,B
    #A=np.arange(1,m*n+1)
    #np.random.shuffle(A)
    #A=A.reshape((m,n))
    if not flag:
        A=np.random.randn(n)*n+np.arange(1,n+1)
        B=np.random.randn(n)*n+np.arange(1,n+1)
        print repr(A),repr(B)
        flag=True  
        A=np.diag(A)
        B=np.diag(B)
              
    #B=np.arange(1,m*n+1)+m*n*np.random.randn(m*n)
    ##B=np.arange(1,m*n+1)
    #np.random.shuffle(B)
    #B=B.reshape((n,m))
    
    sam_size=1
    tr_exp=0
    mat=0
    for i in range(sam_size):
        a=haar_measure(n/2)
        U=np.vstack((np.hstack((a.real,a.imag)),np.hstack((-a.imag,a.real))))#.reshape((a.shape[0]*2,-1))
        #tr_exp+=np.trace(A.dot(U).dot(B).dot(B.T).dot(U.T).dot(A.T))/float(n)-tr_exp-np.trace(A.dot(A.T))/float(m)-np.trace(B.dot(B.T))/float(n)
        #mat+=A.T.dot(A).dot(U).dot(B).dot(B.T).dot(U.T)-A.T.dot(A).dot(B).dot(B.T)
        tr_exp+=np.trace(A.T.dot(A).dot(U).dot(B).dot(B.T).dot(U.T))/float(n)-np.trace(A.dot(A.T))/float(n)*np.trace(B.dot(B.T))/float(n)
    #print mat/float(sam_size)
    #a=haar_measure(m*n/2)
    #U=np.vstack((np.hstack((a.real,a.imag)),np.hstack((-a.imag,a.real))))#.reshape((a.shape[0]*2,-1))
    #A_U=A.flatten().dot(U).reshape(A.shape)
    #tr_exp=np.trace(A_U.dot(B).dot(B.T).dot(A_U.T))/float(n)-tr_exp-np.trace(A.dot(A.T))/float(m)-np.trace(B.dot(B.T))/float(n)
    
    tr_bad=np.trace(A.T.dot(A).dot(B).dot(B.T))/float(n)-np.trace(A.dot(A.T))/float(n)*np.trace(B.dot(B.T))/float(n)
    return tr_exp/float(np.linalg.norm(A.dot(A.T),2)*np.linalg.norm(B.dot(B.T),2)),tr_bad/(float(np.linalg.norm(A.dot(A.T),2)*np.linalg.norm(B.dot(B.T),2)))
    #return tr_exp/float(sam_size),tr_bad
    #return tr_exp/(float(sam_size)*np.linalg.norm(A.dot(A.T),2)*np.linalg.norm(B.dot(B.T),2)),tr_bad/np.linalg.norm(A.dot(A.T),2)*np.linalg.norm(B.dot(B.T),2)
#print trace_check(10,10)
trials=100
res=[]
for i in range(trials):
    res.append(trace_check(50))
res=np.asarray(res).reshape((-1,2))
#print np.mean(res[:,0]<0)
print res
print np.mean(np.abs(res[:,0])<np.abs(res[:,1]))
print haar_measure(5)

    
