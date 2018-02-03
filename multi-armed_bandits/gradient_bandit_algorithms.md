# 2.8 梯度老虎机算法

到目前为止，在本章中，我们已经考虑了估计行为价值的方法，并将这些估计用于选择行为。这是一个好方法，但并不是唯一的。在本节中，我们考虑学习每个行为a的偏好，我们用$H_t(a)$表示。偏好越大，采取行动的次数就越多，但是偏好在奖励方面没有任何解释。只有一个行动相对于另一个行动的相对偏好是重要的；如果我们将所有的偏好加1000，那么这对根据soft-max分布确定的动作概率没有影响。soft-max分布（比如Gibbs or Boltzmann分布）如下：
$$
Pr\lbrace A_r = a\rbrace \doteq \cfrac{e^{H_T(a)}}{\sum_{b=1}^k{e^{H_t(b)}}} \doteq \pi_t(a)
$$
图2.4：10臂测试平台上的ucb动作选择的平均表现。如图所示，ucb通常比$\varepsilon$-贪婪动作选择更好，除了在前k个步骤，当它在未经实验的动作中随机选择时。

在这里我们也介绍了一个有用的新符号，$\pi_t(a)$，表示在t时刻采取行动a的概率。最初所有的偏好都是相同的（例如，对于所有的a，$H_1(a)=0$），使得所有的动作具有相等的被选择的概率。

**练习2.7** 在两个行为的条件下，soft-max分布与统计和人工神经网络中经常使用的logistic或sigmoid函数给出的相同。

基于随机梯度上升的思想，存在一种自然的学习算法。在每个步骤中，在选择动作$A_t$并接收奖励$R_t$之后，偏好被更新为：
$$
H_{t+1}(A_t)\doteq H_t(A_t)+\alpha(R_t-\overline{R_t})(1-\pi_t(A_t))\hspace{1cm} \text{and} \\
H_{t+1}(a)\doteq H_t(a)-\alpha(R_t-\overline{R_t})\pi_t(a) \hspace{1cm} \text{for all } a \ne A_t
$$
其中$\alpha>0$是一个步长参数，$\overline{R_t}\in\mathbb{R}$是包括时间t在内的所有奖励的平均值，可以按照2.4节（或者2.5节，如果问题是非平稳的）中所描述的递增计算。$\overline{R_t}$项作为比较奖励的基线。如果奖励高于基准，那么未来获得的概率增加，如果奖励低于基准，则概率降低。未被选中的动作会朝相反的方向移动。

图2.5显示了梯度老虎机算法在10臂测试平台的一个变体中的结果，其中真实的预期奖励的正态分布的均值是+4而不是零（方差还和原来一样是单位方差）。所有奖励的增加对于梯度老虎机算法是完全没有影响的，因为奖励基线项瞬时适应新的水平。但是如果基线被忽略（也就是说，如果（2.10）中的$\overline{R_t}$被认为是恒定的零），那么性能将显著降低，如图所示。

图2.5：当$q_*(a)$被选择为接近+4而不是接近零时，在10臂测试平台上有和没有奖励基线的梯度bandit算法的平均性能。

**梯度老虎机算法的随机梯度上升**

可以通过将其理解为对梯度上升的随机逼近来获得对梯度老虎机算法的更深入的了解。在确切的梯度上升中，每个偏好$H_t(a)$将与增量对表现的影响成比例递增：
$$
H_{t+1}(a)\doteq H_t(a)+\alpha \cfrac{\partial \mathbb{E}[R_t]}{\partial H_t(a)}
$$
这里表现用预计的奖励来衡量：
$$
\mathbb{E}[R_t]=\sum_b \pi_t(b)q_*(b)
$$
增量影响的衡量是这种表现衡量偏好的偏导数。当然，在我们的情况下不可能实现梯度上升，因为我们不能通过假设知道$q_*(b)$，但实际上我们的算法（2.10）的更新等于（2.11）的期望值，使这个算法成为随机梯度上升的一个实例。计算显示这只需要微积分，但需要几个步骤。首先我们仔细看看推导的表现梯度：
$$
\cfrac{\partial \mathbb{E}[R_t]}{\partial H_t(a)}= \cfrac{\partial}{\partial H_t(a)}[\sum_b\pi_t(b)q_*(b)] \\
 = \sum_bq_*(b)\cfrac{\partial\pi_t(b)}{\partial H_t(a)} \\
 = \sum_b(q_*(b)-X_t)\cfrac{\partial\pi_t(b)}{\partial H_t(a)}
$$
其中$X_t$可以是任何不依赖于$b$的标量。我们可以在这里包含它，因为在所有的行为中，梯度总和为零，$\sum_b\cfrac{\partial\pi_t(b)}{\partial H_t(a)}$。随着$H_t(a)$的变化，一些行为的概率上升，一些下降，但是这些变化的总和必须是零，因为概率之和必须保持为1。
$$
\cfrac{\partial\mathbb{E}[R_t]}{\partial H_t(a)}=\sum_b\pi_t(b)(q_*(b)-X_t)\cfrac{\partial\pi_t(b)}{\partial H_t(a)}/\pi_t(b)
$$
该方程现在是一个期望的形式，将随机变量的所有可能值b相加，然后乘以取这些值的概率。也就是：
$$
=\mathbb{E}[(q_*(A_t)-X_t)\cfrac{\partial\pi_t(A_t)}{\partial H_t(a)}/\pi_t(A_t)] \\
=\mathbb{E}[(R_t-\overline{R_t})\cfrac{\partial\pi_t(A_t)}{\partial H_t(a)}/\pi_t(A_t)]
$$
在这里我们选择了$X_t=\overline{R_t}$，并用$q_*(A_t)$替代$R_t$，这是允许的，因为$\mathbb{E}[R_t|A_t]=q_*(A_t)$，而且$R_t$（给出$A_t$）与其他任何东西都不相关。不久我们将建立$\cfrac{\partial\pi_t(b)}{\partial H_t(a)}=\pi_t(b)(\mathbb{1}_{a=b}-\pi_t(a)$，其中$\mathbb{1}_{a=b}$被定义为如果a = b，值为1，否则为0。基于现在的假设，我们就有
$$
=\mathbb{E}[(R_t-\overline{R_t})\pi_t(A_t)(\mathbb{1}_{a=A_t}-\pi_t(a))/\pi_t(A_t)] \\
=\mathbb{E}[(R_t-\overline{R_t})(\mathbb{1}_{a=A_t}-\pi_t(a))]
$$
回想一下，我们的计划是把性能梯度写成我们可以在每一步中进行抽样的东西，就像我们刚才所做的那样，在每一步中都迭代地按比例进行更新。（2.11）中的性能梯度代入上述期望的样本得到：
$$
H_t(a)=H_t(a)+\alpha(R_t-\overline{R_t})(\mathbb{1}_{a=A_t}-\pi_t(a))\text{,     for all a}
$$
你可能会认为它等同于我们原来的算法（2.10）。

正如我们所假设的那样，它表明$\cfrac{\partial\pi_t(b)}{\partial H_t(a)}=\pi_t(b)(\mathbb{1}_{a=b}-\pi_t(a))$。回想一下导数的标准商准则：
$$
\cfrac{\partial}{\partial x}[\cfrac{f(x)}{g(x)}]=\cfrac{\cfrac{\partial f(x)}{\partial x}g(x)-f(x)\cfrac{\partial g(x)}{\partial x}}{g(x)^2}
$$
使用这个，就有
$$
\begin{equation}
\begin{split}
\frac{\partial \pi_t(b)}{\partial H_t(a)} &= \frac{\partial}{\partial H_t(a)}\pi_t(b) \\
&= \frac{\partial}{\partial H_t(a)}[\frac{e^{H_t(b)}}{\sum^k_{c=1}e^{H_t(c)}}] \\
&= \frac{\frac{\partial e^{H_t(b)}}{\partial H_t(a)}\sum^k_{c=1}e^{H_t(c)}-e^{H_t(b)}\frac{\partial \sum^k_{c=1}e^{H_t(c)}}{\partial _t(a)}}{(\sum^k_{c=1}e^{H_t(c)})^2}   \quad  \text{（商数定理）}\\
&= \frac{\mathbb{1}_{a=b}e^{H_t(b)}\sum^k_{c=1}e^{H_t(c)}-e^{H_t(b)}e^{H_t(a)}}{(\sum^k_{c=1}e^{H_t(c)})^2} \quad \text{（因为}\frac{\partial e^x}{\partial x}=e^x \text{）}\\
&= \frac{\mathbb{1}_{a=b}e^{H_t(b)}}{\sum^k_{c=1}e^{H_t(c)}}-\frac{e^{H_t(b)}e^{H_t(a)}}{(\sum^k_{c=1}e^{H_t(c)})^2}\\
&= \mathbb{1}_{a=b}\pi_t(b)-\pi_t(b)\pi_t(a)\\
&= \pi_t(b)(\mathbb{1}_{a=b}-\pi_t(a)) \quad  \text{ 证明完毕}
\end{split}
\end{equation}
$$
我们刚刚表明梯度老虎机算法的预期更新等于期望奖励的梯度，因此这个算法是随机梯度算法的一个实例。这保证了算法具有鲁棒的收敛性。

请注意，我们不需要任何奖励基线的属性，它不依赖于选择的行为。例如，我们可以将基线设置为零或1000，算法仍然是随机梯度上升的实例。基线的选择并不影响算法的预期更新，但它确实会影响更新的方差，因此会影响收敛速度（如图2.5展示的那样）。选择平均奖励作为基线可能不是最好的，但它很简单，在实践中运作良好。