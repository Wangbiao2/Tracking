# High-Speed Tracking with Kernelized Correlation Filters

+ 利用循环偏移构建循环矩阵
+ 将问题求解变换到傅里叶域
+ 避免了矩阵求逆，降低了算法复杂度

## 0.算法流程

![KCF算法流程](C:\Users\86156\Desktop\KCF算法流程.png)

## 1.岭回归求解

训练样本集$(x_i,y_i)$，线性回归函数$f(x_i)=w^Tx_i$，岭回归最小二乘：
$$
\min _w \sum_i\left(f\left(x_i\right)-y_i\right)^2+\lambda\|w\|^2
$$
线性回归的最小二乘解：
$$
w=\left(X^H X+\lambda I\right)^{-1} X^H y
$$

## 2.循环矩阵

KCF的训练样本是由目标样本通过循环移位得到的。
$$
\boldsymbol{X}=\boldsymbol{C}(\vec{x})=\left[\begin{array}{ccccc}
x_1 & x_2 & x_3 & \ldots & x_n \\
x_n & x_1 & x_2 & \ldots & x_{n-1} \\
x_{n-1} & x_n & x_1 & \ldots & x_{n-2} \\
\ldots & \ldots & \ldots & \ldots & \ldots \\
x_2 & x_3 & x_4 & \ldots & x_1
\end{array}\right]
$$
$\vec{x}$表示基样本，对其循环移位$n$次($\boldsymbol{C}(\vec{x})$)，得到循环矩阵$\boldsymbol{X}$

+ __任何循环矩阵可以被傅里叶变换矩阵（酉矩阵）对角化：__

$$
\boldsymbol{X}=\boldsymbol{C}(x)=\boldsymbol{F} \operatorname{diag}(\widehat{x}) \boldsymbol{F}^H
$$

+ $\widehat{x}$表示$\vec{x}$的离散傅里叶变换，$\widehat{x}=\mathcal{F}(\vec{x})=\sqrt{n} \boldsymbol{F} x$

## 3.线性回归训练提速 

代入得
$$
\begin{aligned}
w & =\left(\boldsymbol{X}^H \boldsymbol{X}+\lambda \boldsymbol{I}\right)^{-1} \boldsymbol{X}^H \vec{y} \\
& =\left(\boldsymbol{F} \operatorname{diag}\left(\widehat{x}^* \odot \widehat{x}\right) \boldsymbol{F}^H+\lambda \boldsymbol{F} \boldsymbol{I} \boldsymbol{F}^H\right)^{-1} \boldsymbol{X}^H \vec{y} \\
& =\left(\boldsymbol{F} \operatorname{diag}\left(\widehat{x}^* \odot \widehat{x}\right) \boldsymbol{F}^H+\boldsymbol{F} \operatorname{diag}(\lambda) \boldsymbol{F}^H\right)^{-1} \boldsymbol{X}^H \vec{y} \\
& =\left(\boldsymbol{F} \operatorname{diag}\left(\widehat{x}^* \odot \widehat{x}+\lambda\right) \boldsymbol{F}^H\right)^{-1} \boldsymbol{X}^H \vec{y} \\
& =\left[\left(\boldsymbol{F}^H\right)^{-1} \operatorname{diag}\left(\widehat{x}^* \odot \widehat{x}+\lambda\right)^{-1} \boldsymbol{F}^{-1}\right] \boldsymbol{X}^H \vec{y} \\
& =\left[\boldsymbol{F} \operatorname{diag}\left(\frac{1}{\widehat{x}^* \odot \widehat{x}+\lambda}\right) \boldsymbol{F}^{-1}\right] \boldsymbol{X}^H \vec{y}
\end{aligned}
$$

$$
\begin{aligned}
w  & =F \operatorname{diag}\left(\frac{1}{\hat{x} \odot \hat{x}^*+\lambda}\right) F^H F \operatorname{diag}\left(\hat{x}^*\right) F^H \vec{y} \\
& =F \operatorname{diag}\left(\frac{\hat{x}^*}{\hat{x} \odot \hat{x}^*+\lambda}\right) F^H \vec{y}
\end{aligned}
$$

+ __反用对角化性质：$F \operatorname{diag}(x) F^H=C\left(\mathcal{F}^{-1}(x)\right)$，上式等号右边前三项仍构成一个循环矩阵__，所以

$$
w=C\left(\mathcal{F}^{-1}\left(\frac{\hat{x}^*}{\hat{x} \odot \hat{x}^*+\lambda}\right)\right) \vec{y}
$$

+ __利用循环矩阵卷积性质，==$\mathcal{F}(C(x) \cdot y)=\hat{x}^* \odot \hat{y}$==__，得
  $$
  \begin{aligned}
  \mathcal{F}(w) & =\left(\frac{\hat{x}^*}{\hat{x} \odot \hat{x}^*+\lambda}\right)^* \odot \mathcal{F}(\vec{y}) \\
  & =\frac{\hat{x}}{\hat{x} \odot \hat{x}^*+\lambda} \odot \mathcal{F}(\vec{y})\\
  & =\frac{\hat{x} \odot \hat{y}}{\hat{x} \odot \hat{x}^*+\lambda}
  \end{aligned}
  $$

即:
$$
\hat{w}=\frac{\hat{x} \odot \hat{y}}{\hat{x} \odot \hat{x}^*+\lambda}
$$

+ $w$可以通过离散傅里叶变换和对位乘法的带，点积运算取代矩阵运算，同时避开矩阵求逆。

## 4.非线性回归训练提速

+ 找到一个__非线性映射函数__$\varphi(x)$，使映射后得样本在新空间中线性可分，在新空间中就可以使用岭回归来寻找一个分类器：$$f\left(\boldsymbol{x}_i\right)=\boldsymbol{w}^T \varphi\left(\boldsymbol{x}_i\right)$$

+ 将线性滤波器的解$w$用样本的线性组合来表示：

$$
\boldsymbol{w}=\sum_i \alpha_i \varphi\left(\boldsymbol{x}_i\right)
$$

+ 优化问题则是求解$\alpha$

+ 线性条件下的回归问题，经过非线性变换后为：

$$
\begin{aligned}
f(z) & =\boldsymbol{w}^T \boldsymbol{z} \\
& =\left(\sum_i^n \alpha_i \varphi\left(\boldsymbol{x}_i\right)\right)^T \cdot \varphi(\boldsymbol{z}) \\
& =\sum_i^n \alpha_i \varphi^T\left(\boldsymbol{x}_i\right) \varphi(\boldsymbol{z}) \\
& =\sum_i^n \alpha_i \mathcal{K}\left(\boldsymbol{x}_i, \boldsymbol{z}\right)
\end{aligned}
$$

+ 预测：新样本与所有旧样本内积的加权平均。
+ 核矩阵$K$是所有训练样本的核相关矩阵，$K_{i j}=\mathcal{K}\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)$

+ __核函数下岭回归的解：__

$$
\boldsymbol{\alpha}=(K+\lambda I)^{-1} \boldsymbol{y}
$$

+ 定理：给定循环数据$C(x)$，对于任意的变换矩阵$M$，如果核函数$\mathcal{K}\left(\boldsymbol{x}, \boldsymbol{x'}\right)=\mathcal{K}\left(\boldsymbol{Mx}, \boldsymbol{Mx'}\right)$，则核矩阵$K$是循环矩阵

+ 径向基函数核、点积核、加权核都满足这一定理
+ __$\alpha$化简__

​				==设核矩阵$K$的初始向量为$k^{xx}$，则：==
$$
\begin{aligned}
\alpha & =\left(C\left(k^{x x}\right)+\lambda I\right)^{-1} y \\
& =\left(F \operatorname{diag}\left(\hat{k}^{x x}\right) F^H+\lambda I\right)^{-1} y \\
& =\left(F \operatorname{diag}\left(\hat{k}^{x x}+\lambda\right) F^H\right)^{-1} y \\
& =F \operatorname{diag}\left(\frac{1}{\hat{k}^{x x}+\lambda}\right) F^H y
\end{aligned}
$$

+ 傅里叶变换

$$
\begin{gathered}
F^H \alpha=\operatorname{diag}\left(\frac{1}{\hat{k}^{x x}+\lambda}\right) F^H y \\
\hat{\alpha}^*=\operatorname{diag}\left(\frac{1}{\hat{k}^{x x}+\lambda}\right) \hat{y}^* \\
\hat{\alpha}=\frac{\hat{y}}{\hat{k}^{x x}+\lambda}
\end{gathered}
$$

+ 结果对比

$$
\begin{gathered}
原闭式解：\alpha=(K+\lambda I)^{-1} y \\
简化结果：\hat{\alpha}=\frac{\hat{y}}{\hat{k}^{x x}+\lambda}
\end{gathered}
$$

## 5.核相关矩阵的加速计算

### 5.1.点积与多项式核

+ 点积核函数$g$可表示为：

$$
\mathcal{K}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=g\left(\boldsymbol{x}^T \boldsymbol{x}^{\prime}\right)
$$

$$
\begin{gathered}
k_i^{\boldsymbol{x} x^{\prime}}=\mathcal{K}\left(\boldsymbol{x}^{\prime}, P^{i-1} \boldsymbol{x}\right)=g\left(\boldsymbol{x}^{\prime} P^{i-1} \boldsymbol{x}\right) \\

\end{gathered}
$$

$$
k^{\boldsymbol{x} \boldsymbol{x}^{\prime}}=g\left(C(\boldsymbol{x}) \boldsymbol{x}^{\prime}\right)
$$

由循环矩阵性质可知：
$$
\begin{aligned}
& \mathcal{F}\left(C(\boldsymbol{x}) \boldsymbol{x}^{\prime}\right)=\widehat{\boldsymbol{x}}^* \odot \boldsymbol{x}^{\prime} \\
\Rightarrow & C(\boldsymbol{x}) \boldsymbol{x}^{\prime}=\mathcal{F}^{-1}\left(\widehat{\boldsymbol{x}}^* \odot \widehat{\boldsymbol{x}}^{\prime}\right) \\
\Rightarrow & k^{\boldsymbol{x} \boldsymbol{x}^{\prime}}=g\left(\mathcal{F}^{-1}\left(\widehat{\boldsymbol{x}}^* \odot \widehat{\boldsymbol{x}}^{\prime}\right)\right)
\end{aligned}
$$

+ 特别的，对于多项式核$\mathcal{K}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=\left(\boldsymbol{x}^T \boldsymbol{x}^{\prime}+a\right)^b$

$$
k^{\boldsymbol{x} x^{\prime}}=\left(\mathcal{F}^{-1}\left(\widehat{\boldsymbol{x}}^* \odot \widehat{\boldsymbol{x}}^{\prime}\right)+a\right)^b
$$

### 5.2.径向基函数与高斯核

+ 径向基核函数$h$可表示为：

$$
\mathcal{K}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=h\left(\left\|\boldsymbol{x}-\boldsymbol{x}^{\prime}\right\|^2\right)
$$

$$
\begin{aligned}
k_i^{\boldsymbol{x} x^{\prime}}=\mathcal{K}\left(\boldsymbol{x}^{\prime}, P^{i-1} \boldsymbol{x}\right) & =h\left(\left\|\boldsymbol{x}^{\prime}-P^{i-1} \boldsymbol{x}\right\|^2\right) \\
& =h\left(\left\|\boldsymbol{x}^{\prime}\right\|^2+\left\|P^{i-1} \boldsymbol{x}\right\|^2-2 \boldsymbol{x}^{\prime T} P^{i-1} \boldsymbol{x}\right) \\
& =h\left(\left\|\boldsymbol{x}^{\prime}\right\|^2+\|\boldsymbol{x}\|^2-2 \boldsymbol{x}^{\prime T} P^{i-1} \boldsymbol{x}\right) \\
& =h\left(\left\|\boldsymbol{x}^{\prime}\right\|^2+\|\boldsymbol{x}\|^2-2 \mathcal{F}^{-1}\left(\widehat{\boldsymbol{x}}^* \odot \widehat{\boldsymbol{x}}^{\prime}\right)\right)
\end{aligned}
$$

+ 特别的，对于高斯核函数

$$
\mathcal{K}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=\exp \left(-\frac{1}{\sigma^2}\left\|\boldsymbol{x}-\boldsymbol{x}^{\prime}\right\|^2\right)
$$

$$
k^{x x^{\prime}}=\exp \left(-\frac{1}{\sigma^2}\left(\left\|\boldsymbol{x}^{\prime}\right\|^2+\|\boldsymbol{x}\|^2-2 \mathcal{F}^{-1}\left(\widehat{\boldsymbol{x}}^* \odot \widehat{\boldsymbol{x}}^{\prime}\right)\right)\right)
$$

## 6.快速检测

$$
f(z)=K\alpha\\
小范围搜索，评估多个候选区域，仍采用循环矩阵\\
=(K^z)^T\alpha
$$

其中：

+ $\alpha$为训练好的分类器参数
+ $K^z=C\left(\boldsymbol{k}^{\boldsymbol{x} z}\right)=\mathcal{K}\left(P^{i-1} \boldsymbol{z}, P^{j-1} \boldsymbol{x}\right)$，表示**训练样本和待检测样本之间的核矩阵**，是一个非对称矩阵
+ $x$表示待训练基样本，$z$表示待检测基样本

$$
\begin{aligned}
f(\boldsymbol{z}) & =\left[C\left(\boldsymbol{k}^{\boldsymbol{x z}}\right)\right]^T \boldsymbol{\alpha} \\
& =F \operatorname{diag}\left(\left(\widehat{\boldsymbol{k}}^{\boldsymbol{x} z}\right)^*\right) F^H \boldsymbol{\alpha} \\
& =C\left(\left(\boldsymbol{k}^{\boldsymbol{x} z}\right)^*\right) \boldsymbol{\alpha}
\end{aligned}
$$

上式用了循环矩阵的转置性质：$X^T=F \operatorname{diag}\left((\widehat{\boldsymbol{x}})^*\right) F^H$

+ 傅里叶变换

$$
\mathcal{F}(f(\boldsymbol{z}))=\mathcal{F}\left(C\left(\left(\boldsymbol{k}^{\boldsymbol{x} z}\right)^*\right) \boldsymbol{\alpha}\right)
$$

+ 利用循环矩阵性质

$$
\begin{aligned}
\mathcal{F}(f(\boldsymbol{z})) & =\mathcal{F}^*\left(\left(\boldsymbol{k}^{\boldsymbol{x z}}\right)^*\right) \odot \mathcal{F}(\boldsymbol{\alpha}) \\
& =\mathcal{F}\left(\boldsymbol{k}^{\boldsymbol{x} z}\right) \odot \mathcal{F}(\boldsymbol{\alpha})
\end{aligned}
$$

+ 即：

$$
\hat{f(z)}=\hat{k_{xz}}\odot\hat{\alpha}
$$

## 7.KCF算法总结

1. 将非线性回归引入到高维空间，转换为线性可分问题，利用岭回归模型对滤波器进行训练，寻找一个函数
   $$
   f\left(\boldsymbol{x}_i\right)=\boldsymbol{w}^T \varphi\left(\boldsymbol{x}_i\right)
   $$
   使样本与目标标签之间的平方误差最小，用于滤波器训练的回归方程为：
   $$
   \varepsilon(\boldsymbol{w})=\left\|\sum_{c=1}^D \boldsymbol{w}^{c *} \varphi\left(\boldsymbol{x}^c\right)-\boldsymbol{y}\right\|^2+\lambda \sum_{c=1}^D\left\|\boldsymbol{w}^c\right\|^2
   $$

2. 利用核技巧将$w$映射到高维空间，设$\boldsymbol{w}=\sum_i \alpha_i \varphi\left(\boldsymbol{x}_i\right)$，引入循环矩阵特性，简化$\alpha$的解为：
   $$
   \hat{\boldsymbol{\alpha}}=\frac{\hat{\boldsymbol{y}}}{\hat{\boldsymbol{k}}^{x x}+\lambda \boldsymbol{I}_N}
   $$

3. 求解$\alpha$之前，需要求训练样本的核相关矩阵，两个单特征通道向量和两个多特征通道向量的多项式与高斯核相关矩阵的计算方式：
   $$
   单通道\left\{\begin{array}{l}
   \boldsymbol{k}_{\mathrm{d}}^{\mathrm{xx}^{\prime}}=\left(\boldsymbol{F}^{-1}\left(\sum \hat{\boldsymbol{x}}^* \odot \hat{\boldsymbol{x}}^{\prime}\right)+\boldsymbol{a}\right)^b \\
   \boldsymbol{k}_{\mathrm{g}}^{\boldsymbol{xx}^{\prime}}=\exp \left(-\frac{1}{\sigma^2}\left(\|\boldsymbol{x}\|^2+\left\|\boldsymbol{x}^{\prime}\right\|^2-2 \boldsymbol{F}^{-1}\left(\sum \hat{\boldsymbol{x}}^* \odot \hat{\boldsymbol{x}}^{\prime}\right)\right)\right)
   \end{array}\right.
   $$

$$
多通道\left\{\begin{array}{l}
\boldsymbol{k}_{\mathrm{d}}^{\mathrm{xx}^{\prime}}=\left(\boldsymbol{F}^{-1}\left(\sum_c \hat{\boldsymbol{x}}_c^* \odot \hat{\boldsymbol{x}}_c^{\prime}\right)+\boldsymbol{a}\right)^b \\
\boldsymbol{k}_{\mathrm{g}}^{\mathrm{xx}^{\prime}}=\exp \left(-\frac{1}{\sigma^2}\left(\|\boldsymbol{x}\|^2+\left\|\boldsymbol{x}^{\prime}\right\|^2-2 \boldsymbol{F}^{-1}\left(\sum_c \hat{\boldsymbol{x}}_c^* \odot \hat{\boldsymbol{x}}_c^{\prime}\right)\right)\right)
\end{array}\right.
$$

4. 通过更新滤波器参数$\alpha$和样本$x$对当前帧模型进行更新

$$
\left\{\begin{array}{l}
\hat{\boldsymbol{\alpha}}_{f, \text { model }}=(\mathbf{1}-\eta) \hat{\boldsymbol{\alpha}}_{f-1, \text { model }}+\eta \hat{\boldsymbol{\alpha}}_f \\
\boldsymbol{x}_{f, \text { model }}=(\mathbf{1}-\eta) \boldsymbol{x}_{f-1, \text { moxde1 }}+\eta \boldsymbol{x}_f
\end{array}\right.
$$

5. 利用更新后的参数进行检测，得到响应函数

$$
f(z)=F^{-1}\left(\sum_c \hat{\boldsymbol{k}}^{x z} \odot \hat{\boldsymbol{\alpha}}\right)
$$

6. 根据响应函数，推导出响应最大值点，最大值点对应的偏移量即为检测位置。
7. 根据响应最大值点的信息重新训练滤波器，更新模型。

## 8. KCF不足

![image-20230219165205844](image-20230219165205844.png)

## 9.评估指标

1. 距离精度（计算中心定位误差）

$$
\mathrm{DP}=\frac{N_{\mathrm{CIEsth_{0 }}}}{N} \times 100 \%
$$

2. 重叠精度（计算重叠得分）

$$
\mathrm{OP}=\frac{N_{0 \mathrm{~s} \geqslant \mathrm{th}_1}}{N} \times 100 \%
$$

3. 时序鲁棒性评估（选择不同的起始帧）

4. 空间鲁棒性评估（标定不同的目标边界框）

5. 精度图（显示了中心定位误差在不同给定阈值下，满足条件的视频帧占视频总帧的百分比）
6. 成功图（显示了重叠得分在不同给定阈值( 0 到 1 之间) 下，满足条件的视频帧占视频总帧的百分比）

## 10. SiamFC 全卷积孪生网络跟踪算法

+ 分别用相同的网络对样本图像和搜索图像进行特征提取，再对二者的特征图进行互相关操作，生成响应图，响应图最高的位置就最有可能是跟踪目标的位置。

![image-20230219162243374](C:\Users\86156\AppData\Roaming\Typora\typora-user-images\image-20230219162243374.png)

+ __UAV123数据集__测试：

<img src="C:\Users\86156\AppData\Roaming\Typora\typora-user-images\image-20230219162506730.png" alt="image-20230219162506730" style="zoom: 33%;" />

<img src="D:\Pytorch_wangbiao\SiamFC_huanglianghua\tools\reports\UAV123\SiamFC\precision_plots.png" alt="precision_plots" style="zoom: 25%;" />

<img src="success_plots.png" alt="success_plots" style="zoom:25%;" />

+ __OTB100数据集测试__

  ![0001](0001.jpg)

<img src="precision_plots.png" alt="precision_plots" style="zoom:25%;" />

<img src="D:/Pytorch_wangbiao/SiamFC_huanglianghua/OTB_Result/reports/OTB2015/SiamFC/success_plots.png" alt="success_plots" style="zoom:25%;" />

KCF算法（2014）对快速尺度变化跟踪效果非常不理想，SiamFC（2016）跟踪效果很好，但是基于相关滤波的方法速度快一些。

+ SiamFC效果

<img src="image-20230219164207884.png" alt="image-20230219164207884" style="zoom: 50%;" />

+ KCF效果（也有[这个代码](https://github.com/chuanqi305/KCF.git)可能没有加窗（循环移位导致时域截断，频谱泄露）的因素）

<img src="image-20230219164347723.png" alt="image-20230219164347723" style="zoom: 50%;" />

+ MOSSE（2010）效果也可以

<img src="image-20230219164740082.png" alt="image-20230219164740082" style="zoom:50%;" />