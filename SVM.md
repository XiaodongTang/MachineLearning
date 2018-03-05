## SVM

支持向量机，提出于1995年bell-lab。适用于二分类问题，可解决线性可分和线性不可分问题。原理很简单，求解两个分类的最大几何间距，在数学上该问题是一个凸二次规划最优解问题，可根据拉格朗日对偶性将其转化成对引入的拉格朗日因子的求解问题。如果线性不可分，一种解决方法是对异常点加入松弛变量，把类别仍然当作线性可分，另一种是引入核函数。最后使用smo方法对以上问题进行求解。

推导过程参考这篇[文章](http://blog.csdn.net/alwaystry/article/details/60957096),建议手动画画，加深理解。

### SVM小尝试


### SVM解决多分类问题

Hsu C W, Lin C J. A comparison of methods for multiclass support vector machines[J]. IEEE Transactions on Neural Networks, 2002, 13(4):1026.




