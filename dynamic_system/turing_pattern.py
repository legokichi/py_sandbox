# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# 偏微分方程式のシミュレーション
# フィッツフュー・南雲方程式で表される反応拡散系

# 次の偏微分方程式で表される系の時間発展を $E=[-1, 1]^2$ の領域でシミュレート
#$$
#      \frac{\partial{u}}{\partial{t}} = a \Delta u + u - u^3 -v + k \\
# \tau \frac{\partial{u}}{\partial{t}} = b \Delta v + u - v
#$$
# 変数uは皮膚の色素沈着を促進する物質の密度
# 変数vは変数uと反応し色素沈着を阻害する物質の密度

# 初期状態ではuとvはランダムで独立な分布と仮定
# ノイマン境界条件=領域の境界において空間導関数はnull（離散的？ぼかしがない？

# モデルのパラメータを設定
a = 2.8e-4
b = 5e-3
tau = .1
k = -.005


# 時空間を離散化
# 離散化するときに安定しているための条件
# dt \leq \frac{dx^2}{2}
size = 80 # 空間のグリッドサイズ
dx = 2./size # 空間のステップ
T = 10.0 # 総時間
dt = .9 * dx**2/2 # 時間ステップ
n = int(T/dt)/2/2


# 2次元格子の頂点におけるそれぞれの物質の変数値、範囲[0,1]でランダム
U = np.random.rand(size, size)
V = np.random.rand(size, size)


# 5点ステンシル有限差分法(なにそれ)
# 2次元格子上の変数に対するラプラス作用素を計算する関数
#$$
# \Delta u(x, y) \simeq \frac{
#    u(x+h,y) + u(x-h,y)
#  + u(x,y+h) + u(x,y-h)
#  - 4u(x, y)}{dx^2}
#
#$$

def laplacian(Z):
    # 格子の境界をとりさる
    Ztop    = Z[0:-2, 1:-1]
    Zleft   = Z[1:-1, 0:-2]
    Zbottom = Z[2:  , 1:-1]
    Zright  = Z[1:-1, 2:  ]
    Zcentor = Z[1:-1, 1:-1]
    return (Ztop + Zleft + Zbottom + Zright - 4 * Zcentor) / dx**2



# 有限差分法を使う
for i in range(n):
    # 各ステップにおいて2つの方程式の右辺を離散化ラプラシアンを使って計算
    deltaU = laplacian(U)
    deltaV = laplacian(V)
    # 格子内部のuとvの値を取り出す
    Uc = U[1:-1, 1:-1]
    Vc = V[1:-1, 1:-1]
    # 変数を更新
    U[1:-1, 1:-1], V[1:-1, 1:-1] = (
        Uc + dt * (a*deltaU + Uc - Uc**3 - Vc + k),
        Vc + dt * (b*deltaV + Uc - Vc) / tau
    )
    # ノイマン境界条件、境界値をnullにする
    for Z in (U, V):
        Z[ 0, :] = Z[ 1, :]
        Z[-1, :] = Z[-2, :]
        Z[ :, 0] = Z[ :, 1]
        Z[ :,-1] = Z[ :,-2]



fig = plt.figure()
plt.imshow(U, cmap=plt.cm.copper, extent=[-1,1,-1,1])
plt.show()
