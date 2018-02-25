import numpy as np

# x = np.random.randint(1,20,[7,3])
# y = np.random.randint(1,3,[3,3])

x = np.array([[0,1],[1,1],[2,1]]).reshape(3,2)

# x = np.column_stack(x0,x1)
print(x)

y = np.array([2,4,6]).reshape(3,1)

# y = np.random.uniform(1,20,7).reshape(7,1)
# z = np.random.standard_normal(10)
# z = np.random.uniform(1,5,10)
def min_2_mult(x,y):    
    p1 = x.T
    p2=p1.dot(x)
    p3=np.linalg.inv(p2)
    p4=p3.dot(p1)
    p5=p4.dot(y)
    return p5


print("\n")

print(min_2_mult(x,y))





