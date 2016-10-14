import numpy as np
import chainer.functions as F
from chainer import optimizers
from chainer import Variable
from chainer import FunctionSet

n_units = 10
model = FunctionSet(
    l1=F.Linear(5, n_units),
    l2=F.Linear(n_units, n_units),
    l3=F.Linear(n_units, 5))
optimizer = optimizers.SGD()
optimizer.setup(model)
def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)), train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    y = model.l3(h2)
    return F.mean_squared_error(y, t), y
x = Variable(np.array([[ 1, 2, 3, 4, 5]], dtype=np.float32))
t = Variable(np.array([[11,12,13,14,15]], dtype=np.float32))
for i in range(0, 1000):
    optimizer.zero_grads()
    loss, y = forward(x.data, t.data)
    loss.backward()
    optimizer.update()
    loss, y = forward(x.data, t.data, False)
    print("loss:", loss.data)
    print("y:", y.data)
x = Variable(np.array([[6, 7, 8, 9, 10]], dtype=np.float32))
loss, y = forward(x.data, t.data, False)
print("loss:", loss.data)
