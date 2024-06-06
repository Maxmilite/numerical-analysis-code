## 第一章作业上机实验

### > 求方程 $x^2 + (\alpha - \beta) x + 10^9 = 0$ 的根。其中 $\alpha = -10^9, \beta = -1$，讨论如何设计计算格式才能有效减少误差，提高运算精度。

由于待求解方程组为一元二次方程组，因此可直接使用求根公式 $$x = \cfrac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$ 求解。在题设背景中，$a = 1, b = \alpha + \beta = -10^9 - 1, c = 10^9$。

注意到 $|b| = \sqrt{b^2 - 4ac}$，且 $b < 0$，因此应用改进形式：
$$
x = \cfrac{-2c}{b \pm \sqrt{b^2 - 4ac}}
$$

解得 $x_1 = 10^9$，$x_2 = 1$。Python 求解代码如下：

```python
alpha, beta = -1e9, -1
a, b, c = 1, alpha + beta, 1e9

x1, x2 = (-2 * c) / (b + (b ** 2 - 4 * a * c) ** 0.5) \
       , (-2 * c) / (b - (b ** 2 - 4 * a * c) ** 0.5)

print(x1, x2)
```

### > 以计算 $x^{31}$ 为例，讨论如何设计计算格式才能减少计算次数。

注意到例如 $31 = 2^0 + 2^1 + 2^2 + 2^3 + 2^4$，任何正整数均有唯一一个二进制拆分方式。

令 $f_i = x^{2^i}$，注意到 $f_i = f_{i - 1} \times f_{i - 1}$。因此可以在将指数做二进制分解的同时，通过迭代方式计算 $f_i$。从而在 $\mathcal{O}(\log x)$ 时间复杂度内计算 $a^x$（$a = \text{const}$），远优于朴素乘法的 $\mathcal{O}(x)$。

例如，对 $x^{31}$，可分解为 $x^{2^0} \times x^{2^1} \times x^{2^2} \times x^{2^3} \times x^{2^4}$ 的形式。

假设 $x = 42$，求得 $x^31 = 209280132851277227501988209234026802122409700753408$。Python 求解代码如下：

```python
def fpow(x: float, y: int) -> float:
    res: float = 1
    while y:
        if y & 1:
            res *= x
        x *= x
        y >>= 1
    return res

x, y = 42, 31
print(fpow(x, y))
```

## 第二章作业上机实验

### > 求方程 $2x^2 + x - 15 = 0$ 的正根（$x* = 2.5$）近似值。

$x_{k+1} = 15 - x_k^2$ 不收敛，因数值过大无法计算。

$x_{k + 1} = \cfrac{15}{2x_k + 1}$ 结果如图：

![](figures/fig1.png)

$x_{k + 1} = x_k - \cfrac{2x_k^2 + x_k - 15}{4x_k + 1}$ 结果如图：

![](figures/fig2.png)

**收敛性分析**


#### 1. $x_{k+1} = 15 - x_k^2$
这个公式基于将方程变形为：
$ 2x^2 + x - 15 = 0 \implies x = 15 - x^2 $

**稳定性和收敛性分析：**
- 在根 $x^* = 2.5$ 附近，设 $x_k = x^* + \epsilon_k$，其中 $\epsilon_k$ 是小的偏差。
- 代入迭代公式：
  $ x_{k+1} = 15 - (x^* + \epsilon_k)^2 $
  $ x_{k+1} = 15 - (2.5 + \epsilon_k)^2 $
  $ x_{k+1} = 15 - (6.25 + 5\epsilon_k + \epsilon_k^2) $
  $ x_{k+1} = 8.75 - 5\epsilon_k - \epsilon_k^2 $

- 对于小的$\epsilon_k$，忽略$\epsilon_k^2$项：
  $ x_{k+1} \approx 8.75 - 5\epsilon_k $
  这里我们看到，误差$\epsilon_k$被放大了5倍，所以这个迭代公式是不稳定的并且是发散的。

#### 2. $x_{k+1} = \frac{15}{2x_k + 1}$
这个公式基于将方程变形为：
$ 2x^2 + x - 15 = 0 \implies x = \frac{15}{2x + 1} $

**稳定性和收敛性分析：**
- 在根 $x^* = 2.5$ 附近，设 $x_k = x^* + \epsilon_k$，其中 $\epsilon_k$ 是小的偏差。
- 代入迭代公式：
  $ x_{k+1} = \frac{15}{2(x^* + \epsilon_k) + 1} $
  $ x_{k+1} = \frac{15}{2(2.5 + \epsilon_k) + 1} $
  $ x_{k+1} = \frac{15}{6 + 2\epsilon_k} $
  $ x_{k+1} \approx \frac{15}{6(1 + \frac{2\epsilon_k}{6})} $
  $ x_{k+1} \approx \frac{15}{6} \cdot \left(1 - \frac{2\epsilon_k}{6}\right) $
  $ x_{k+1} \approx 2.5 \cdot \left(1 - \frac{\epsilon_k}{3}\right) $
  $ x_{k+1} \approx 2.5 - \frac{2.5\epsilon_k}{3} $
  $ x_{k+1} \approx 2.5 - 0.833\epsilon_k $

- 这里误差$\epsilon_k$被缩小了约0.833倍，所以这个迭代公式是收敛的，但收敛速度较慢。

#### 3. $x_{k+1} = x_k - \frac{2x_k^2 + x_k - 15}{4x_k + 1}$
这个公式基于牛顿法：
$ f(x) = 2x^2 + x - 15 $
$ f'(x) = 4x + 1 $
$ x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)} = x_k - \frac{2x_k^2 + x_k - 15}{4x_k + 1} $

**稳定性和收敛性分析：**
- 在根 $x^* = 2.5$ 附近，设 $x_k = x^* + \epsilon_k$，其中 $\epsilon_k$ 是小的偏差。
- 牛顿法通常具有二阶收敛速度：
  $ x_{k+1} = x_k - \frac{f(x^* + \epsilon_k)}{f'(x^* + \epsilon_k)} $

  由于牛顿法的二阶收敛性质：
  $ \epsilon_{k+1} \approx C\epsilon_k^2 $
  其中C是一个常数。

- 因此，误差的平方使得收敛速度非常快。这表明这个迭代公式在根附近收敛速度非常快，通常是最有效的方法。

#### 总结
- $x_{k+1} = 15 - x_k^2$：不稳定，发散。
- $x_{k+1} = \frac{15}{2x_k + 1}$：稳定，收敛速度较慢。
- $x_{k+1} = x_k - \frac{2x_k^2 + x_k - 15}{4x_k + 1}$：稳定，收敛速度快（二阶收敛）。

### > 证明方程 $2 - 3x - \sin x = 0$ 在 $(0, 1)$ 内有且仅有一个实根。使用二分法求误差不大于 $0.0005$ 的根，及需要的迭代次数。

令 $f(x) = 2 - 3x - \sin x$，有 $f'(x) = -3 - \cos x < 0$，因此其在定义域内单调递减。又 $f(0) = 2$，$f(1) = -1 - \sin 1 < 0$，故 $f(x) = 0$ 在 $(0, 1)$ 内只有一个实根。

在使用二分法迭代约 $11$ 次后，解得 $x = 0.50537109375$，此时 $f(x) = 0.0016466850065690042$，且误差在 $0.0005$ 以内。

Python 代码如下：

```python
def f(x: float) -> float:
    return 2 - 3 * x - math.sin(x)

TOL = 0.0005
l, r = 0, 1
cnt = 0
while (r - l) > TOL:
    cnt += 1
    x = (l + r) / 2
    if f(x) > 0:
        l = x
    else:
        r = x

print(x, cnt, f(x))
```

### > 利用牛顿法求解方程
> $$ \cfrac 12 + \cfrac 14 x^2 - x \sin x - \cfrac 12 \cos 2x = 0$$
> 分别取 $x_0 = \cfrac \pi 2, 5 \pi, 10 \pi$ 使得精度不小于 $10^{-5}$。比较初值对计算结果的影响。

结果如下，其中 `res` 代表最终的结果，`cnt` 代表迭代次数：

```plain
x0 = 1.5707963267948966, res = 1.8954884189769137, cnt = 15
x0 = 15.707963267948966, res = 1.8954890013731998, cnt = 19
x0 = 31.41592653589793, res = 1.8955029706849522, cnt = 428
```
初值对结果的影响是，虽然最终结果是相同的，但是迭代次数可能会有所不同。如果初值过大，可能会导致迭代次数过多，甚至不收敛。

### > 已知 $f(x) = 5x - e^x$ 在 $(0, 1)$ 之间有一个实根，试分别利用二分法、牛顿法、割线法、错位法设计相关的计算格式，并编程求解。精确到四位小数。

在二分法中，取二分点为 $x = \frac{l + r}{2}$，在牛顿法中，取初始点为 $x_0 = 0.5$，在割线法中，取初始点为 $x_0 = 0, x_1 = 1$，在错位法中，取初始点为 $x_0 = 0, x_1 = 1$，结果如下：

```plain
bisection: res = 0.99993896484375, cnt = 14
newton: res = 0.25917110166149104, cnt = 3
secant: res = 0.2591712288816814, cnt = 4
regula_falsi: res = 0.25917408510579704, cnt = 4
```

Python 代码如下：

```python
def f(x: float) -> float:
    return 5 * x - math.exp(x)

def df(x: float) -> float:
    return 5 - math.exp(x)

def bisection(l: float, r: float) -> float:
    TOL = 1e-4
    cnt = 0
    while (r - l) > TOL:
        cnt += 1
        x = (l + r) / 2
        if f(x) > 0:
            l = x
        else:
            r = x
    return x, cnt

def newton(x0: float) -> float:
    TOL = 1e-4
    cnt = 0
    while True:
        cnt += 1
        x = x0 - f(x0) / df(x0)
        if abs(x - x0) < TOL:
            break
        x0 = x
    return x, cnt

def secant(x0: float, x1: float) -> float:
    TOL = 1e-4
    cnt = 0
    while True:
        cnt += 1
        x = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x - x1) < TOL:
            break
        x0, x1 = x1, x
    return x, cnt

def regula_falsi(x0: float, x1: float) -> float:
    TOL = 1e-4
    cnt = 0
    while True:
        cnt += 1
        x = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x - x1) < TOL:
            break
        if f(x) * f(x1) < 0:
            x0 = x
        else:
            x1 = x
    return x, cnt

l, r = 0, 1
res, cnt = bisection(l, r)
print(f'bisection: res = {res}, cnt = {cnt}')

x0 = 0.5
res, cnt = newton(x0)
print(f'newton: res = {res}, cnt = {cnt}')

x0, x1 = 0, 1
res, cnt = secant(x0, x1)
print(f'secant: res = {res}, cnt = {cnt}')

x0, x1 = 0, 1
res, cnt = regula_falsi(x0, x1)
print(f'regula_falsi: res = {res}, cnt = {cnt}')
```

## 第三章作业上机实验

### > 基于不同边界条件的样条函数计算公式推导：自然边界、固定边界、周期边界、强制第一个子区间和第二个子区间样条多项式的三阶导数相同，倒数第二个子区间和最后一个子区间的三次样条函数的三阶导数相等。

#### 自然边界

在自然边界条件下，要求在端点处的二阶导数为零。这意味着样条函数的曲率在边界上为零，常用于边界不受约束的情况。

对于一个在节点 $(x_0, y_0), (x_1, y_1), \cdots, (x_n, y_n)$ 上的三次样条函数 $S(x) $，可以表示为：

$ S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3 ~ \text{for} \; x \in [x_i, x_{i+1}] $

自然边界条件：$ S''(x_0) = 0, S''(x_n) = 0 $

##### 推导

对于自然边界条件，需要解以下线性方程组：

$ h_i = x_{i+1} - x_i$，$\Delta y_i = y_{i+1} - y_i $

构建样条函数 $ S_i(x) $ 的系统：

$ S_i''(x_{i+1}) = S_{i+1}''(x_{i+1})$，$\int_{x_i}^{x_{i+1}} S_i''(x) \, dx = \Delta y_i - \frac{h_i}{6}(c_i + 2c_{i+1})$，$S''(x_0) = 0$，$S''(x_n) = 0 $

最终会得到一个三对角矩阵的线性方程组，可以用来解 $ c_i $，然后反过来求 $ a_i $，$ b_i $ 和 $ d_i $。

#### 固定边界

在固定边界条件下，要求在端点处的一阶导数（斜率）给定。设端点处的斜率为 $ f'(x_0) $ 和 $ f'(x_n) $。

固定边界条件：$ S'(x_0) = f'(x_0), S'(x_n) = f'(x_n) $

##### 推导

对于固定边界条件，构建样条函数 $ S_i(x) $ 的系统：

$ S_i(x_i) = y_i$，$S_i(x_{i+1}) = y_{i+1}$，$S_i'(x_i) = f'(x_i)$，$S_i'(x_{i+1}) = f'(x_{i+1}) $

#### 周期边界

周期边界条件用于周期性数据，这时要求样条函数及其导数在首尾相接处一致。

周期边界条件：$ S(x_0) = S(x_n)$，$S'(x_0) = S'(x_n)$，$S''(x_0) = S''(x_n) $

##### 推导

对于周期边界条件，需要解以下线性方程组：

$ S_0(x_0) = S_{n-1}(x_n)$，$S_0'(x_0) = S_{n-1}'(x_n)$，$S_0''(x_0) = S_{n-1}''(x_n) $

并结合内部节点的连续性条件来形成完整的方程组。

#### 第一个子区间和第二个子区间样条多项式的三阶导数相等，倒数第二个子区间和最后一个子区间的三次样条函数的三阶导数相等

$ S'''_0(x) = S'''_1(x) ~ \text{for} \; x \in [x_0, x_1]$，$S'''_{n-1}(x) = S'''_n(x) ~ \text{for} \; x \in [x_{n-1}, x_n] $

### > 以 $y = \sin(x)$ 为例，在 $[0,π]$ 区间内生成 $11$ 个、$21$ 个数据点，设计算法或程序，用上述 $4$ 个边界条件，分别计算其样条插值，并作图比较，分析其差异性。

