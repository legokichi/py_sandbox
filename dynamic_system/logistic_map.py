# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

def logistic(r, x):
    return r*x*(1 - x)


# 2.5から4の間のn個の値でパラメータを変えてシミュレート
n = 10000 # 10000個の並列世界を作る
r = np.linspace(2.5, 4.0, n) # r[0],r[1]...ごとにロジスティック写像のパラメータが異なる世界、系
# r = [2.5,  2.51515152,  2.53030303, ...]
# 力学系のステップ回数
iterations = 1000 # 1000時間後の世界を観察
last = 100

# 初期値 x_0 = 0.00001
x = 1e-5*np.ones(n) # すべての世界の初期値は同一

# リアプノフ指数の初期化
lyapunov = np.zeros(n) # = [0,0,...]

# 分岐図の表示
plt.subplot(111)
for i in range(iterations):
    x = logistic(r, x) # 1ステップ時間を進める
    lyapunov += np.log(abs(r-2*r*x)) # リアプノフ指数の部分合計を計算
    # r-2*r*x は ロジスティック写像 rx(1-x)=rx-rx^2 のxによる導関数 r-2rx
    if i >= (iterations - last):
        plt.plot(r, x, ',k', alpha=.02) # ラスト100を分岐図に書き込んでいく
plt.xlim(2.5, 4)
plt.title("bifurcation diagram") # 分岐図
# 横軸 rが異なる世界の種類
# 縦軸 最後100回の世界の値
plt.show()

# リアプノフ指数（初期値鋭敏性の指標)
plt.subplot(111)
print(lyapunov, r)
print(r[lyapunov<0], lyapunov[lyapunov<0 ])
plt.plot(r[lyapunov<0],  lyapunov[lyapunov<0 ]/iterations, ',k', alpha=0.1)
plt.plot(r[lyapunov>=0], lyapunov[lyapunov>=0]/iterations, ',r', alpha=0.25)
plt.xlim(2.5, 4)
plt.ylim(-2, 1)
plt.title("Lyapunov exponent")
plt.show()
