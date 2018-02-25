import numpy as np
import matplotlib.pyplot as plt

def f1(t):
    return np.exp(-t)*np.cos(2*np.pi*t)

p1 = plt.subplot(222)
p2 = plt.subplot(224)

#从X中随机选择k个数据
def randomChoice(X,k):
    # print(X.shape[0])
    if X.shape[0] < k:
        return 
    x_id = np.random.choice(X.shape[0],size=k,replace=False)
    print(x_id)
    x_rand = X[x_id,:] 
    return x_rand 


# def drawPicture(X，color):
#     x = X[:,0]
#     y = X[:,1]
#     p1.plot(x,y,color)

def K_Means(X,k):         
    initial_centroids = randomChoice(X,k)  # 初始化类中心
    print("随机初始化中心：","\n",initial_centroids)
    max_iters = 3                          #最大循环次数
    d ,s = runKMeans(X,initial_centroids,max_iters)   # 执行K-Means聚类算法
    print("类中心：","\n",d)
    print("数据","\n",X)
    print("类别：","\n",s)

    x = X[:,0]
    y = X[:,1]
    p1.plot(x,y,"g.")
   
# 找到每条数据距离哪个类中心最近    
def findClosestCentroids(X,initial_centroids):
    m = X.shape[0]                  # 数据条数
    K = initial_centroids.shape[0]  # 类的总数
    dis = np.zeros((m,K))           # 存储计算每个点分别到K个类的距离
    idx = np.zeros((m,1))           # 要返回的每条数据属于哪个类
    
    #计算每个点到每个类中心的距离
    for i in range(m):
        for j in range(K):
            dis[i,j] = np.dot((X[i,:]-initial_centroids[j,:]).reshape(1,-1),(X[i,:]-initial_centroids[j,:]).reshape(-1,1))    
   
    dummy,idx = np.where(dis == np.min(dis, axis=1).reshape(-1,1))
    return idx[0:dis.shape[0]]  

# 计算类中心
def computerCentroids(X,idx,K):
    n = X.shape[1] #维数
    centroids = np.zeros((K,n))
    for i in range(K):
        centroids[i,:] = np.mean(X[np.ravel(idx==i),:], axis=0).reshape(1,-1)   # 索引要是一维的,axis=0为每一列，idx==i一次找出属于哪一类的，然后计算均值
    return centroids

# 聚类算法
def runKMeans(X,initial_centroids,max_iters):
    m,n = X.shape                   # 数据条数和维度
    K = initial_centroids.shape[0]  # 类数
    centroids = initial_centroids   # 记录当前类中心
    idx = np.zeros((m,1))           # 每条数据属于哪个类
    
    for i in range(max_iters):      # 迭代次数       
        idx = findClosestCentroids(X, centroids)       
        centroids = computerCentroids(X, idx, K)    # 重新计算类中心       
    
    return centroids,idx    # 返回聚类中心和数据属于哪个类


x1 = np.arange(0,5,1)
x2 = np.arange(20,25,1)
x3 = np.arange(40,45,1)
x4 = np.arange(60,65,1)
x5 = np.arange(80,85,1)

x0 = np.append(x1,x2)
x0 = np.append(x0,x3)
x0 = np.append(x0,x4)
x0 = np.append(x0,x5)
x0=x0.reshape(25,1)

print(x0)
X = np.array([[1,2],[3,3],[6,2],[8,5],[7,4],[9,1],[12,3]])     
x = X[:,0]
y = X[:,1]

p1.plot(x,y,"g.")

k = 5
K_Means(X,k)

plt.show()
