《Introduction to Machine Learning with Python》《Python机器学习基础教程》

[TOC]

## Supervised Learning

## Unsupervised Learning

## Python Libs(Packages)

### scikit-learn

- **Iris datasets**
- //TODO...
- scikit-learn中的train_test_split函数将75%的行数据及对应标签作为训练集，剩下25%的数据及其标签作为测试集
- 一部分数据用于**构建机器学习模型**，叫作训练数据（training data）或**训练集**（training set）。其余的数据用来**评估模型性能**，叫作测试数据（test data）、**测试集**（test set）或留出集（hold-out set）
- scikit-learn中的**数据**通常用**大写的X**表示，而**标签**用**小写的y**表示
- 用大写的X是因为数据是一个二维数组（矩阵），用小写的y是因为目标是一个一维数组（向量），这也是数学中的约定

#### 	depends on NumPy and SciPy

#### Numpy(key point)

#### Scipy

#### matplotlib(key point)

#### pandas

#### mglearn [本书github代码](https://github.com/amueller/introduction_to_ml_with_python)

## Chapter2 Supervised Learning 监督学习

### 2.1 分类（classifification）和回归（regression）

#### #分类

- 分类问题的目标是**预测类别标签**（class label），这些标签来自**预定义的可选列表**

- 分类问题有时可分为**二分类** （binary classification，在两个类别之间进行区分的

  一种特殊情况）和**多分类**（multiclass  classification，在两个以上的类别之间进行区分）

- **二分类**（寻找垃圾邮件）中，通常一个类别称为**正类**（positive class），另一个类别称为**反类**（negative class）

- **多分类**（鸢尾花，根据网站上的文本预测网站所用的语言。这里的类别就是预定义的语言表）

#### #回归

- 回归任务的目标是**预测一个连续值**，编程术语叫作**浮点数**（floating-point number），数学术语叫作**实数**（real number）
- 举例：
  - 根据教育水平、年龄和居住地来预测一个人的**年收入**
  - 根据上一年的产量、天气和农场员工数等属性来预测玉米农场的**产量**
  - 预测值可以**在给定范围内任意取值**

#### #区分分类和回归

- **输出是否具有某种连续性**（ some kind of ordering or continuity in the output）

### 2.2 泛化、过拟合与欠拟合

- 如果一个模型能够对没见过的数据做出准确预测，我们就说它能够从**训练集泛化**（generalize）到**测试集**。我们想要构建一个泛化精度尽可能高的模型。

- **通常**来说，我们构建模型，使其在**训练集**上能够做出准确**预测**。如果训练集和测试集足够**相似**，我们预计模型在**测试集**上也能做出**准确预测**。

- 判断一个算法在新数据上表现好坏的**唯一度量**，就是在**测试集**上的评估，测试集是用来测试模型**泛化性能**的数据

- **简单**的模型对新数据的**泛化**能力**更好**

- 模型**过于复杂**，被称为**过拟合**（**overfitting**）,在拟合模型时**过分关注训练集的细节**，得到了一个在**训练集上表现很好**(预测效果好)、但不能泛化到新数据上的模型

- 模型**过于简单**，模型甚至在训练集上的表现就很差，选择过于简单的模型被称为**欠拟合**（**underfitting**）

- 二者之间存在一个**最佳位置**，可以得到最好的泛化性能，过拟合与欠拟合之间的权衡：

  ![](.\ML_COURSE\模型复杂度与训练精度和测试精度之间的权衡.png)

#### 模型复杂度与数据集大小的关系

- 模型复杂度与训练数据集中输入的变化密切相关,数据集中包含的数据点的**变化范围越大**，在不发生过拟合的前提下你可以使用的**模型就越复杂**,收集**更多数据**，**适当构建**更复杂的模型，对监督学习任务往往特别有用

### 2.3监督学习算法

#### 2.3.1 一些样本数据集

使用低维数据集：（**特征较少，可视化简单**），结论不适用于高维数据集（特征较多）

- **模拟**的forge二分类数据集，绘制散点图，**第一个特征**为x轴，**第二个特征**为y轴

  `mglearn.discrete_scatter(X[:, 0], X[:, 1], y)`

  ```python
  Out[2]:
  X.shape: (26, 2) #这个数据集包含26个数据点和2个特征
  ```

- **模拟**的wave数据集说明回归算法，wave 数据集只有

  一个输入特征和一个连续的目标变量（或响应），后者是模型想要预测的对象。**单一特征**位于x轴，**回归目标**（输出）位于y轴。

  现实数据集包含在scikit-learn中,通常被保存为Bunch对象,可以用点操作符来访问对象的值（比

  如用**bunch.key**来代替bunch['key']）。

- **现实**世界的cancer数据集，肿瘤（良性：benign,恶性：malignant）,其任务是基于人体组织的测量数据来**学习预测**肿瘤是否为**恶性**。

  ```python
  from sklearn.datasets import load_breast_cancer
  cancer = load_breast_cancer()
  ```

  `cancer.data.shape: (569,30)` 569个数据点，每个数据点有30个特征，**具体详见书本介绍**。

- **现实**世界的**回归**数据集，即波士顿房价数据集。任务：利用犯罪率、是否邻近查尔斯河、公路可达性等信息，来预测20世纪70年代波士顿地区房屋价格的中位数。

  ```python
  from sklearn.datasets import load_bostonboston = load_boston()
  ...
  ```

  `Data shape: (506, 13)`

  我们不仅将犯罪率和公路可达性作为特征，还将

  犯罪率和公路可达性的**乘积**作为特征。像这样包含导出特征的方法叫作**特征工程**（feature 

  engineering）,将在第4章中详细讲述。

#### 2.3.2 k近邻 k-Nearest Neighbor

- 最简单的机器学习算法

- 构建模型只需要保存**训练数据集**即可。想要

  对新数据点做出**预测**，算法会在**训练数据集中**找到最近的数据点，也就是它的“**最近邻**”。

  1. **k近邻分类**

     - k-NN算法最简单的版本只考虑一个最近邻（**单一最近邻算法**）：对于**每个新数据点**，标记了**训练集中**与它**最近**的点。

     - 还可以考虑任意个（**k个**）邻居，用“**投票法**”（voting）来指定标签。就是说，对

       于**每个测试点**，数一数多少个邻居属于类别0，多少个邻居属于类别1。然后将出现

       **次数更多的类别**（也就是k个近邻中占多数的类别）作为**预测结果**。（看图）同样适用于多分类数据集。

       ```python
       clf = KNeighborsClassifier(n_neighbors=3) #实例化类
       clf.fit(X_train, y_train) #利用训练集对分类器进行拟合
       print("Test set predictions: {}".format(clf.predict(X_test)))#调用predict方法来对测试数据进行预测
       print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))#评估模型的泛化能力好坏，对测试数据和测试标签调用score方法
       ```

  2. **分析KNeighborsClassifier**

     根据平面中每个点所属的类别对平面进行着色。这样可以查看**决策边界**（decision boundary），即算法对类别0和类别1的**分界线**。（决策边界可视化，**见书**）：随着邻居个数越来越多，决策边界也越来越**平滑**。更平滑的边界对应**更简单**的模型。换句话说，**使用更少的邻居对应更高的模型复杂度，而使用更多的邻居对应更低的模型复杂度**。

     书本图2-7：

     ​	**邻居数越多，模型越简单，训练集精度下降，测试集精度大体先升后降。**

  3. **k近邻回归** 

     ...**//todo**

     - 利用单一邻居的预测结果就是最近邻的目标值
   - 在使用多个近邻时，预测结果为这些邻居的平均值
  
4. 分析KNeighborsRegressor
  
   ...
  
5. **优点、缺点和参数**
  
   一般来说，KNeighbors分类器有**2个**重要参数：**邻居个数**与**数据点之间距离的度量方法**。
  
   在实践中，使用**较小**的邻居个数（比如3个或5个）往往可以得到**比较好**的结果，但你应
  
   该调节这个参数。**默认**使用**欧式距离**，它在
  
     许多情况下的效果都**很好**。
  
     k-NN的优点之一就是模型很**容易理解**，但如果训练集**很大**（特征数很多或者样本数很大），预测速度可能会**比较慢**。使用k-NN算法时，对数据进行**预处理**是**很重要**的（见第3章）。对于大多数特征的大多数取值都为0的数据集（所谓的**稀疏数据集**）来说，这一算法的效果**尤其不好**。（kNN在在实践中往往不会用到，下面的这种方法就没有这两个缺点）

#### 2.3.3  线性模型 Linear models

线性模型利用输入特征的线性函数（linear function）进行预测

1. **用于回归的线性模型**

   - 对于回归问题，线性模型预测的一般公式如下：

     ***ŷ* = *w*[0] * *x*[0] + *w*[1] * *x*[1] + … \+ *w*[*p*] * *x*[*p*] + *b***

   *x*[0] 到 *x*[*p*] 表示单个数据点的**特征**（本例中特征个数为 *p*+1），*w* 和 *b* 是学习模型的 

   **参数**，*ŷ* 是模型的**预测结果**

   - 对于**单一**特征的数据集，公式如下：

     ***ŷ* = *w*[0] * *x*[0] + *b***
     
     在一维 wave 数据集上学习参数 *w*[0] 和 *b*：
     
     ```python
     In[25]:
     mglearn.plots.plot_linear_regression_wave()
     Out[25]:
     w[0]: 0.393906 b: -0.031804
     ```
     
     用于回归的线性模型可以表示为这样的回归模型：对单一特征的预测结果是一条**直线**，两个特征时是一个**平面**，或者在更高维度（即更多特征）时是一个**超平面**。

2. **线性回归（又名普通最小二乘法）**（ordinary least squares，OLS）,是**回归问题**最简单也最经

   典的线性方法

   - 线性回归寻找参数w和b，使得对训练集的预测值与真实的回归目标值y之间的**均方误差最小**

   - **均方误差（mean squared error）**是预测值与真实值之差的平方和除以样本数
     $$
     (预测值-真实值)^2/样本数
     $$

   - 线性回归没有参数，这是一个优点，但也因此无法控制模型的复杂度

   - 线性模型对**wave数据集**的预测结果（书P36 37）

     ```python
     from sklearn.linear_model import LinearRegressionX, 
     y = mglearn.datasets.make_wave(n_samples=60)
     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
     ...
     #“斜率”参数（w，也叫作权重或系数）被保存在coef_属性中，而偏移或截距（b）被保存在intercept_属性中：
     #intercept_属性是一个浮点数，而coef_属性是一个NumPy数组
     ```

     ```python
     print("Training set score: {:.2f}".format(lr.score(X_train, y_train))) 0.67
     print("Test set score: {:.2f}".format(lr.score(X_test, y_test))) 0.66
     ```

     训练集和测试集上的分数非常接近。这说明可能存在**欠拟合**，而不是过拟合,对于一维数据集来说，过拟合的风险很小

   - 线性模型对**波士顿房价数据集(复杂模型)**的预测结果

     ```python
     X, y = mglearn.datasets.load_extended_boston()
     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
     lr = LinearRegression().fit(X_train, y_train)
     ```

     ```python
     print("Training set score: {:.2f}".format(lr.score(X_train, y_train))) 0.95
     print("Test set score: {:.2f}".format(lr.score(X_test, y_test))) 0.61
     ```

     测试集预测比训练集**低**很多（**过拟合**），因此应该试图找到一个可以**控制复杂度**的模型。**标准线性回归**最常用的**替代**方法之一就是**岭回归（ridge regression）**

3. **岭回归**

   - 预测公式与**普通最小二乘法**相同

   - 对系数（w）的选择不仅要在训练数据上得到好的预测结果，而且还要拟合附加约束

   - 我们还希望**系数尽量小**。换句话说，w的所有元素都应接近于0。直观上来看，这意味着每个特征对输出的影响应**尽可能小**（即斜率很小），**同时**仍给出很好的预测结果。这种**约束**是所谓**正则化（regularization）**的一个例子

   - 正则化是指对模型做**显式约束**，以**避免过拟合**，岭回归用到的这种被称为**L2正则化**

     ```python
     from sklearn.linear_model import Ridge
     ridge = Ridge().fit(X_train, y_train)
     ```

     ```python
     print("Training set score: {:.2f}".format(ridge.score(X_train, y_train))) 0.89
     print("Test set score: {:.2f}".format(ridge.score(X_test, y_test))) 0.75
     ```

     Ridge在训练集上的分数要低于LinearRegression，但在测试集上的分数更高。**复杂度更小**的模型意味着在**训练集**上的**性能更差**，但**泛化性能更好**

   - **简单性**和**训练集性能**二者对于模型的重要程度可以由用户通过设置**alpha**参数来指定

   - alpha的最佳设定取决于用到的**具体数据集**

   - **增大alpha**会使得**系数更加趋向于0**，从而降低训练集性能，但可能会**提高泛化性能**

     ```python
     ridge10 = Ridge(alpha=10).fit(X_train, y_train)
     ```

   - **减小alpha**可以让系数受到的**限制更小**

     ```python
     ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
     ```

     从数学的观点来看，Ridge**惩罚**了系数的L2范数或w的欧式长度

   - 还有一种方法可以用来理解**正则化的影响**，就是固定alpha值，但**改变训练数据量**，(例:对波士顿房价数据集做**二次抽样**,...将模型性能作为数据集大小的函数进行绘图，这样的图像叫作**学习曲线**

   - //TODO...
   
4. lasso

   - 除了Ridge，还有一种**正则化的线性回归**是Lasso。与岭回归相同，使用lasso也是**约束系数使其接近于0**，但用到的方法不同，叫作**L1正则化**

   - L1正则化的结果是，使用lasso时某些系数**刚好为0**。这说明某些特征被模型完全忽略

   - 与Ridge类似，Lasso也有一个正则化参数**alpha**，可以控

     制**系数趋向于0的强度**。在上一个例子中，我们用的是默认值alpha=1.0。为了**降低**欠拟

     合，我们尝试**减小alpha**。这么做的同时，我们还需要**增加max_iter**的值（运行迭代的最

     大次数）

#### 2.3.4 朴素贝叶斯分类器 Naive Bayes Classifiers

- 训练速度更快，泛化能力比线性分类器稍差

- 朴素贝叶斯模型如此高效的原因在于，它通过**单独**查看**每个特征**来学习参数，并从每个特征中收集简单的类别统计数据

- scikit-learn 中实现了三种朴素贝叶斯分类器：GaussianNB(**高斯**:任意连续数据)、BernoulliNB (**伯努利**:二分类)和 MultinomialNB(**多项式**:假定输入数据为计数数据(每个特征代表某个对象的整数计数))

  - BernoulliNB 分类器计算**每个特征不为0**的元素个数，代码见书P53

  - MultinomialNB

    计算每个类别中**每个特征**的**平均值**

  - GaussianNB

    保存每个类别中**每个特征**的**平均值**和**标准差**

  - **优点、缺点和参数**

    - MultinomialNB和BernoulliNB都只有一个参数**alpha**，用于控制模型复杂度。可以将统计数据“平滑化”（smoothing）。alpha越大，平滑化越强，模型复杂度就越低。
    
    - GaussianNB主要用于**高维数据**，而另外两种朴素贝叶斯模型则广泛用于**稀疏计数数据**，比如文本。MultinomialNB的性能通常要优于BernoulliNB，特别是在包含很多非零特征的数据集（即大型文档）上
    - 该模型对**高维稀疏数据**的效果很好，对参数的鲁棒性也相对较好。朴素贝叶斯模型是很好的基准模型，常用于**非常大**的数据集
  
- 补充：

  **贝叶斯公式**

  <img src=".\ML_COURSE\贝叶斯公式.png" style="zoom:50%;" />

  <img src=".\ML_COURSE\Bayes_Training_data.png" style="zoom:50%;" />

  <img src=".\ML_COURSE\bayes_solution.png" style="zoom:50%;" />

#### 2.3.5 决策树 Decision trees

- 决策树是广泛用于**分类**和**回归**任务的模型

- **构造决策树**

  对数据反复进行**递归划分**，直到划分后的每个区域（决策树的每个**叶结点**）只包含单一目标值（单一类别或单一回归值）。如果树中某个叶结点所包含数据点的目标值都相同，那么这个叶结点就是纯的（pure）

  举例：**two_moons数据集**

- **控制决策树的复杂度**

  - 通常来说，构造决策树直到所有叶结点都是**纯的叶结点**，这会导致模型**非常复杂**，并且对训练数据**高度过拟合**。纯叶结点的存在说明这棵树在**训练集**上的精度是 **100%**

    <img src=".\ML_COURSE\Decision_Tree_1.png" style="zoom:50%;" />

    上图可以看出过拟合，在所有属于类别0的点中间有一块属于类别1的区域。另一方面，有一小条属于类别0的区域，包围着最右侧属于类别0的那个点。这并不是想象中决策边界的样子，这个**决策边界**过于关注**远离**同类别其他点的单个异常点

  - **防止过拟合**的**两种**常见策略

    - 及早停止树的生长，也叫**预剪枝（pre-pruning）**（限制条件可能包括限制树的最大深度(max_depth)、限制叶结点的最大数目(max_leaf_nodes)，或者规定一个结点中数据点的最小数目(min_samples_leaf)来防止继续划分）

    - 先构造树，但随后删除或折叠信息量很少的结点，也叫**后剪枝（post-pruning）**或**剪枝（pruning）**

    - **scikit-learn**的决策树在DecisionTreeRegressor类和DecisionTreeClassifier类中实现。

      scikit-learn**只实现了预剪枝**，没有实现后剪枝

    - 在**乳腺癌(Breast
      Cancer)数据集**查看预剪枝的效果，固定树的random_state（随机数种子：保持不变，则每次随机结果相同，方便对比）

      **默认**展开树(纯叶子结点)

      ```python
      cancer = load_breast_cancer()
      X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
      tree = DecisionTreeClassifier(random_state=0)
      tree.fit(X_train, y_train)
      print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
      print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
      ```
      
      ```python
      Accuracy on training set: 1.000 
      Accuracy on test set: 0.937
      ```
      
      **预剪枝** 
      
      ```python
      tree = DecisionTreeClassifier(max_depth=4, random_state=0)
      tree.fit(X_train, y_train)
      ```
      
      ```python
      Accuracy on training set: 0.988
      Accuracy on test set: 0.951
      ```
      **降低了训练集的精度，提高了测试集的精度**

- **分析决策树**

  - 利用tree模块的export_graphviz函数来将树可视化

- **树的特征重要性（feature importance）**

  - 为每个特征对树的决策的重要性

    进行排序，它都是一个介于0和1之间的数字

  - 如果某个特征的feature_importance_很小，并**不能说明**这个特征没有提供任何信息，这只能说明该特征没有被树选中，可能是因为另一个特征也包含了同样的信息

  - 与线性模型的系数不同，特征重要性始终为**正数**，也**不能说明**该特征对应哪个类别

- **优点、缺点和参数(Strengths, weaknesses and parameters)**

  - 线性模型和回归树对RAM价格数据的预测结果对比

  <img src=".\ML_COURSE\decision_tree VS linear_model prediction.png" style="zoom: 50%;" />

  ​	线性模型对测试数据(2000年以后)给出了很好的预测，不过忽略了训练数据和测试数据中一些更细微的变化。与之相反，树模型完美预测了**训练**数据，但是，一旦输入**超出**了模型训练数据的范围，模型就只能持续预测最后一个已知数据点，树不能在训练数据的范围之外生成“新的”响应。所有基于树的模型都有这个**缺点**

  - 决策树有两个**优点**：
    - 得到的模型很容易可视化，容易理解(non-experted)
    - 算法完全不受**数据缩放**(data scaling)的影响。由于**每个特征**被**单独处理**(processed separately)，而且数据的划分也**不依赖于缩放**，因此决策树算法**不需要特征预处理**，比如归一化(normalization)或标准化(standardization)。**特别是**特征的**尺度**(scales)完全不一样时或者二元特征和连续特征**同时**(mix of binary and continuous features)时，决策树的**效果很好**
  - 决策树的**主要缺点**在于，即使做了预剪枝，它也经常会**过拟合**，**泛化性能很差**。因此，在大多数应用中，往往使用下面介绍的**集成方法**来替代单棵决策树

#### 2.3.6 决策树集成 Ensembles of Decision Trees

​	都以决策树为基础**两种**集成模型：

​	**随机森林（random forest）**和**梯度提升决策树（gradient boosted decision tree）**

##### 1.随机森林 random forest 

- 随机森林背后的**思想**(idea)是，每棵树的预测可能都相对较好，但可能对**部分数据过拟合**。

  如果构造很多树，并且每棵树的预测都很好，但都以不同的方式过拟合，那么我们可以对

​       这些树的结果**取平均值**来**降低过拟合**。既能**减少过拟合又能保持树的预测能力**

- 随机森林中树的**随机化方法**有两种：
  - 通过选择用于构造树的**数据点**
  - 通过选择**每次划分测试的特征**(the features in each split test)

- **构造随机森林**

  - 构造树的个数(**n_estimators参数**)

  - 算法对每棵树进行不同的**随机**选择,以确保树和树之间是有区别的(彼此独立)

  - 想要构造一棵树，首先要对数据进行**自助采样（bootstrap sample）**,从n_samples个数据点中**有放回地**（即同一样本可以被多次抽取）**重复随机抽取**一个样本，共抽取**n_samples次**，有些数据点会缺失(missing)或重复(repeated)

  - 接下来，基于这个**新创建的数据集**来构造决策树，对之前介绍的决策树算法稍作修改：在每个结点处，算法**随机选择特征的一个子集**，并对**其中一个特征**寻找**最佳测试**，而**不是**对每个结点都寻找最佳测试。选择的特征个数由**max_features**参数来控制

  - 两种方法结合保证随机森林

    - **自助采样**：**每棵决策树**的**数据集都是略有不同**

    - 每个结点的**特征选择**：**每棵树**中的**每次划分**都是**基于特征的不同子集**

  - 如果max_features**等于n_features**，那么每次划分都要考虑数据集的**所有特征**，在**特征选择**的过程中**没有**添加**随机性**（不过**自助采样依然存在随机性**）如果设置max_features**等于1**，那么在划分时将无法选择对哪个特征进行测试，只能对随机选择的某个特征搜索不同的阈值。因此，如果max_features**较大**，那么随机森林中的树将会**十分相似**(抽样和选择**重复率高**)，利用**最独特的特征**可以**轻松拟合数据**。如果max_features**较小**，那么随机森林中的树将会**差异很大**，**为了**很好地**拟合**数据，每棵树的**深度都要很大**

  - 利用随机森林进行**预测**，算法首先对森林中的**每棵树**进行预测

    - 对于**回归**问题：可以对这些结果**取平均值**作为最终预测
    - 对于**分类**问题：用到了**“软投票”（soft voting）**策略，每个算法做出“软”预测，给出每个可能的输出标签的**概率**。对所有树的预测概率**取平均值**，然后将**概率最大**的类别作为预测结果

  - **分析随机森林**

    - 将由5棵树组成的随机森林应用到前面研究过的**two_moons数据集**上：

      ```python
      X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
      X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,     random_state=42) #42
      forest = RandomForestClassifier(n_estimators=5, random_state=2) #5棵树
      forest.fit(X_train, y_train)
      ```

    - 每棵树学到的**决策边界**可视化，也将它们的总预测（即整个森林做出的预测）**可视化**：

      <img src=".\ML_COURSE\two_moons decision boundary.png" style="zoom:50%;" />

      5棵树学到的**决策边界大不相同**。每棵树都犯了一些错误，因为这里画出的一些训练点实际上**并没有包含**在这些树的训练集中，原因在于**自助采样**。随机森林比单独每一棵树的过拟合都要小，给出的决策边界也更符合直觉。用到**更多棵树**（通常是几百或上千），从而得到**更平滑**的边界

    - 将包含**100棵树**的随机森林应用在**cancer数据集**上:

      ```python
      forest = RandomForestClassifier(n_estimators=100, random_state=0)
      forest.fit(X_train, y_train)
      print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
      print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
      ```

      ```python
      Accuracy on training set: 1.000
      Accuracy on test set: 0.972
      ```

      在没有调节任何参数的情况下，精度比线性模型或单棵决策树都要**好**

  - 类似地，随机森林也可以给出**特征重要性**（feature importance），计算方法是将森林中**所有树**的**特征重要性**求和并**取平均**，随机森林给出的特征重要性要比单棵树给出的更为**可靠**：

    图见P67

    与单棵树相比，随机森林中有**更多**特征的重要性**不为零**

  - **优点、缺点和参数**

    - 用于**回归和分类的随机森林**是目前应用**最广泛**的机器学习方法之一，**不需要**反复调节参数，**不需要**对数据进行缩放

    - 在大型数据集上构建随机森林可能比较**费时间**，但在一台计算机的多个CPU内核上并行计算也很容易，可以用**n_jobs参数**来调节使用的**内核个数**。使用更多的CPU内核，可以让**速度线性增加**（使用2个内核，随机森林的训练速度会**加倍**）

    - 森林中的**树越多**，它对随机状态选择的**鲁棒性**就越好

    - 对于**维度非常高**的**稀疏数据**（比如**文本数据**），随机森林的表现往往**不是很好**。对于这种

      数据，使用**线性模型可能更合适**，对一个应用来说，如果时间和内存很重要的话，那么**换用**线性模型可能更为明智

    - 调参：

      - n_estimators总是**越大越好**，在你的时间/内存允许的情况下**尽量多**
      - max_features决定每棵树的随机性大小，较小的max_features可以**降低过拟合**,一般来说，好的经验就是使用**默认值**：对于**分类**，默认值是**max_features=sqrt(n_features)**；对于**回归**，默认值是**max_features=log2(n_features)**。增大max_features或max_leaf_nodes有时也可以**提高**性能（improve performance）。它还可以大大降低用于训练和预测的时间和空间要求

##### 2.梯度提升回归树（梯度提升机） gradient boosted regression tree (gradient boosting machines)

- 这个模型既可以用于回归也可以用于分类

- 与随机森林方法不同，梯度提升采用**连续**的方式构造树，每棵树都试图**纠正**前一棵树的错误。**默认**情况下，

  梯度提升回归树中**没有随机化**，而是用到了**强预剪枝**

- 梯度提升树通常使用**深度很小**（1到5之间）的树，这样模型占用的**内存更少**，预测速度也**更快**

- 梯度提升背后的主要思想是合并许多简单的模型（在这个语境中叫作**弱学习器**），比如深

  度较小的树。每棵树只能对部分数据做出好的预测，因此，添加的树越来越多，可以**不断**

  **迭代提高性能**

- 对参数设置更敏感，除了预剪枝与集成中树的数量之外，梯度提升的另一个重要参数是**learning_rate（学习**

  **率）**，用于**控制**每棵树**纠正前一棵树的错误的强度**。较高的学习率意味着每棵树都可以做出较强的修正，这样模型**更为复杂**。通过增大n_estimators来向集成中添加更多树，也可以**增加模型复杂度**，因为模型有**更多机会**纠正训练集上的错误

- 乳腺癌数据集上应用GradientBoostingClassifier的示例。默认使用100棵树，最大深度是3，学习率为0.1

  ```python
  gbrt = GradientBoostingClassifier(random_state=0)
  gbrt.fit(X_train, y_train)
  print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
  print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))
  ```

  ```python
  Accuracy on training set: 1.000
  Accuracy on test set: 0.958
  ```

  由于训练集精度达到100%，所以很可能存在过拟合。为了降低过拟合，我们可以**限制最大深度**来加强预剪枝，也可以**降低学习率**：

  ```python
  # max_depth=1
  gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
  gbrt.fit(X_train, y_train)
  ```

  ```python
  Accuracy on training set: 0.991 
  Accuracy on test set: 0.972
  ```

  ```python
  # learning_rate=0.01
  gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
  gbrt.fit(X_train, y_train)
  ```

  ```python
  Accuracy on training set: 0.988
  Accuracy on test set: 0.965
  ```

  两种方法都降低了训练集精度，在这个例子中，减小树的最大深度**显著提升**了模型性能，而降低学习率仅稍稍提高了泛化性能

- 将**特征重要性**可视化

  <img src=".\ML_COURSE\feature importances gradient boosting cancer.png" style="zoom:50%;" />

  梯度提升树的特征重要性与随机森林的特征重要性有些类似，不过梯度提升**完全忽略了某些特征**

- 由于梯度提升和随机森林两种方法在类似的数据上表现得都很好，因此一种常用的方法就是**先尝试随机森林**，它的**鲁棒性**很好。如果随机森林效果很好，但预测时间**太长**，或者机器学习模型**精度**小数点后第二位的**提高**也很重要，那么**切换成梯度提升**通常会有用

- **优点、缺点和参数**

  - 其**主要缺点**是需要**仔细调参**，而且**训练时间**可能会**比较长**

  - 这一算法**不需要**对数据进行缩放就可以表现得很好，而且**也适用**于二元特征与连续特征同时存在的数

    据集。与其他基于树的模型相同，它也通常**不适用**于**高维稀疏数据**

  - **主要参数**包括**树的数量n_estimators**和**学习率learning_rate**（用于控制每棵树对前一棵树的错误的纠正强度）

  - 这两个参数**高度相关**（highly interconnected），因为learning_rate**越低**，就需要**更多的树**来构建具有相似复杂度的模型

  - 随机森林的n_estimators值总是越大越好，但梯度提升不同，增大n_estimators会导致模型更加复杂，进而可能导致**过拟合**，**通常**的做法是根据**时间和内存**的预算（budget）选择合适的n_estimators，然后对不同的learning_rate进行遍历

  - 另一个**重要参数**是max_depth（或max_leaf_nodes），用于降低每棵树的复杂度。梯度提升模型的max_depth一般不超过5

#### 2.3.7 核支持向量机 Kernelized Support Vector Machines （SVM）

​	//TODO...

1. **线性模型与非线性特征**

   - 让线性模型更加灵活，就是添加更多的特征——举个例子，添加**输入特征的交互项**或**多项式**
   - 现在我们对输入特征进行**扩展**，比如说添加第二个特征的平方（feature1 ** 2）作为一个新特征，将每个数据点表示为**三维点**(feature0, feature1, feature1 ** 2)，而不是二维点(feature0, feature1)

2. **核技巧 （kernel trick）**

   - 向数据表示中添加**非线性特征**，可以让线性模型变得更强大

   - **原理**：直接计算扩展特征表示中数据点之间的距离（更准确地说是**内积**），而不用实际对扩展进行计算
   - 对于支持向量机，将数据**映射到更高维空间**中有**两种**常用的方法：一种是**多项式核**，在一定阶数内计算原始特征所有可能的多项式（比如feature1 ** 2 * feature2 ** 5）；另一种是径向基函数（radial basis function，**RBF**）核，也叫**高斯核**
   - 一种对高斯核的**解释**是它考虑所有阶数的所有可能的多项式，但**阶数越高**，**特征的重要性越小**

3. **理解SVM**

   - 在训练过程中，SVM**学习每个训练数据点**对于**表示两个类别之间的决策边界**的**重要性**。通

     常只有**一部分**训练数据点对于定义决策边界来说很重要：位于**类别之间边界**上的那些**点**。

     这些点叫作**支持向量（support vector）**，支持向量机正是由此得名

4. **SVM调参**

   - 将**RBF核SVM**应用到**乳腺癌数据集**上。默认情况下，C=1，gamma=1/n_features：

     ```python
     X_train, X_test, y_train, y_test = train_test_split(    cancer.data, cancer.target, random_state=0)
     svc = SVC()
     svc.fit(X_train, y_train)
     print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
     print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))
     ```

     ```python
     Accuracy on training set: 1.00
     Accuracy on test set: 0.63
     ```

     训练集上的分数十分完美，但在测试集上的精度只有63%，存在**相当严重的过拟合**

   - SVM对**参数的设定**和**数据的缩放**非常**敏感**。特别地，它要求所有特征有**相似的变化范围**

     查看breast cancer数据集的特征范围：

     <img src=".\ML_COURSE\cancer magnitude.png" style="zoom: 33%;" />

     乳腺癌数据集的特征具有**完全不同的数量级**，这对其他模型来说（比如**线性模型**）可能是**小问题**，但对**核SVM**却有**极大影响**

5. **为SVM预处理数据**

   - 解决这个问题的一种方法就是对**每个特征**进行**缩放**，使其大致都位于同一范围（常用的缩放方法就是将所有特征缩放到0和1之间 **MinMaxScaler**）

   - 手动实现MinMax缩放：

     ```python
     [IN]
     # 计算训练集中每个特征的最小值
     min_on_training = X_train.min(axis=0)
     # 计算训练集中每个特征的范围（最大值-最小值）
     range_on_training = (X_train - min_on_training).max(axis=0)
     # 减去最小值，然后除以范围
     # 这样每个特征都是min=0和max=1
     X_train_scaled = (X_train - min_on_training) / range_on_training
     print("Minimum for each feature\n{}".format(X_train_scaled.min(axis=0)))
     print("Maximum for each feature\n {}".format(X_train_scaled.max(axis=0)))
     ```

     ```python
     [OUT]
     Minimum for each feature[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     Maximum for each feature [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
     ```

     ```python
     [IN]
     # 利用训练集的最小值和范围对测试集做相同的变换（详见第3章）
     X_test_scaled = (X_test - min_on_training) / range_on_training
     svc = SVC()
     svc.fit(X_train_scaled, y_train) # X_train_scaled
     print("Accuracy on training set: {:.3f}".format(svc.score(X_train_scaled, y_train)))
     print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))
     ```

     ```python
     [OUT]
     Accuracy on training set: 0.948 
     Accuracy on test set: 0.951
     ```

     数据缩放的作用很大！实际上模型现在处于**欠拟合**的状态，因为训练集和测试集的**性能非常接近**，但还没有接近100%的精度。从这里开始，我们可以尝试**增大C**或**gamma**来**拟合**更为**复杂**的模型

     ```python
     svc = SVC(C=1000)
     svc.fit(X_train_scaled, y_train)
     ```

     ```python
     Accuracy on training set: 0.988
     Accuracy on test set: 0.972	
     ```

     在这个例子中，增大C可以显著改进模型，得到97.2%的精度

6. **优点、缺点和参数**

   - 它在低维数据和高维数据（即很少特征和很多特征）上的表现都很好，但对样本个数的**缩放表现不好**
   - SVM的另一个**缺点**是，**预处理数据**和**调参**都需要**非常小心**，所以很多应用都使用基于树的模型
   - 此外，SVM模型**很难检查**（inspect），可能很难理解为什么会这么预测，而且也难以将模型向非专家进行解释
   - 不过SVM仍然是值得尝试的，**特别**是所有特征的**测量单位相似**（比如都是**像素密度**）而且**范围也差不多**时
   - 核SVM的**重要参数**是**正则化参数C**、**核的选择**以及与**核相关的参数**
   - 这里主要讲的是**RBF核**，RBF核**只有一个**参数**gamma**，它是**高斯核宽度**的**倒数**
   - gamma和C控制的都是**模型复杂度**，**较大的值**都对应更为**复杂**的模型，因此，这两个参数的设定通常是**强烈相关**的，应该**同时调节**

#### 2.3.8 神经网络（深度学习）Neural Networks (Deep Learning)

这里只讨论一些相对简单的方法，即用于分类和回归的**多层感知机**（multilayer perceptron，MLP），它可以作为研究更复杂的深度学习方法的起点。MLP也被称为（普通）**前馈神经网络**，有时也简称为**神经网络**

##### 1.神经网络模型

- MLP可以被视为**广义的线性模型**，执行**多层处理**后得到结论

- 线性回归的预测公式:

  *ŷ* = *w*[0] * *x*[0] + *w*[1] * *x*[1] + … \+ *w*[*p*] * *x*[*p*] + *b*

  ŷ是输入特征x[0]到x[p]的加权求和，权重为**学到的系数**(coefficients)w[0]到w[p]。

  可以将这个公式**可视化**:

  ```python
  display(mglearn.plots.plot_logistic_regression_graph())
  ```

  <img src=".\ML_COURSE\logistic regression visualization.png" style="zoom:50%;" />

- 在MLP中，**多次重复这个计算加权求和**的过程，首先计算代表**中间过程**的**隐单元（hidden** 

  **unit）**，然后**再计算这些隐单元的加权求和**并得到最终结果(**单隐层的多层感知机**)

  ```python
  display(mglearn.plots.plot_single_hidden_layer_graph())
  ```

  <img src=".\ML_COURSE\single hidden layer.png" style="zoom: 50%;" />

  - 在计算完每个**隐单元的加权求和之后**，**对结果**再应用一个**非线性**函数(**激活函数**（activation functios）)——通常是**校正非线性**（rectifying nonlinearity，也叫校正线性单元或**relu**）或**正切双曲线**（tangens hyperbolicus，**tanh**）。然后将这个函数的结果用于**加权求和**，计算得到输出ŷ

  - 计算回归问题的ŷ的完整公式如下（使用tanh非线性）:

    h[0] = tanh(w[0, 0] * x[0] + w[1, 0] * x[1] + w[2, 0] * x[2] + w[3, 0] * x[3] + b[0])

    h[1] = tanh(w[0, 0] * x[0] + w[1, 0] * x[1] + w[2, 0] * x[2] + w[3, 0] * x[3] + b[1])

    h[2] = tanh(w[0, 0] * x[0] + w[1, 0] * x[1] + w[2, 0] * x[2] + w[3, 0] * x[3] + b[2])

    ŷ = v[0] * h[0] + v[1] * h[1] + v[2] * h[2] + b

    其中，**w**是**输入x与隐层h**之间的**权重**，**v**是**隐层h与输出ŷ** 之间的**权重**。权重w和v要

    从数据中**学习得到**，x是输入特征，ŷ 是计算得到的输出，h是计算的中间结果

- **多隐层的多层感知机-->大型神经网络-->深度学习**

  <img src=".\ML_COURSE\多隐层MLP.png" style="zoom:50%;" />

##### 2.神经网络调参

​	MLPClassifier应用到**two_moons数据集**

```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train) #solver='lbfgs'
...
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
```

<img src=".\ML_COURSE\MLP two_moons.png" style="zoom: 33%;" />

- 包含**100个隐单元**的神经网络在two_moons数据集上学到的**决策边界**

  神经网络学到的决策边界完全是**非线性**(nonlinear)的，但**相对平滑**(relatively smooth)

  **默认**情况下，MLP使用100个隐结点，**默认**的非线性激活函数时**relu**，如果想得到**更加平滑**（模型**更复杂**）的决策边界，可以**添加更多的隐单元**、**添加第二个隐层**或者使用**tanh 非线性**

- 减少hidden units数量（从而**降低**了**模型复杂度**）

  ```python
  # 使用2个隐层，每个包含10个单元，这次使用tanh非线性
  mlp = MLPClassifier(solver='lbfgs', activation='tanh', random_state=0, hidden_layer_sizes=[10, 10])
  mlp.fit(X_train, y_train)
  mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
  mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
  plt.xlabel("Feature 0")
  plt.ylabel("Feature 1")
  ```

  ​				包含2个隐层、每个隐层包含10个隐单元的神经网络学到的决策边界（激活函数为**relu**）

  <img src=".\ML_COURSE\relu.png" style="zoom: 33%;" />

  ​			  包含2个隐层、每个隐层包含10个隐单元的神经网络学到的决策边界（激活函数为**tanh**）

  <img src=".\ML_COURSE\tanh.png" style="zoom: 33%;" />

- 最后，我们还可以利用**L2惩罚**使**权重趋向于0**，从而控制神经网络的**复杂度**，正如我们在

  **岭回归**和**线性分类器**中所做的那样，MLPClassifier中调节L2惩罚的参数是**alpha**（与线

  性回归模型中的**相同**），它的**默认值很小**（**弱正则化**）

  **不同隐单元**个数与**alpha参数**的**不同**设定下的决策函数：

  <img src=".\ML_COURSE\alpha n_hidden .png" style="zoom:50%;" />

- 神经网络的一个重要性质是，在开始学习之前其权重是**随机**设置的，这种随机初始化（随机种子不同）会**影响**（对于较小的网络）学到的模型

- 将MLPClassifier应用在乳腺癌数据集上

  ```python
  mlp = MLPClassifier(random_state=42)
  mlp.fit(X_train, y_train)
  ...
  ```

  ```python
  Accuracy on training set: 0.92 
  Accuracy on test set: 0.90
  ```

  - MLP的精度相当好，但没有其他模型好。与较早的SVC例子相同，原因可能在于数据的**缩放**

  - **神经网络**也要求所有输入特征的**变化范围相似**，最**理想**的情况是**均值(mean)为0**、**方差(variance)为1**

  - 此处手动完成缩放（第3章**StandardScaler**）

    ```python
    # 计算训练集中每个特征的平均值
    mean_on_train = X_train.mean(axis=0)
    # 计算训练集中每个特征的标准差
    std_on_train = X_train.std(axis=0)
    ```

    ```python
    # 减去平均值，然后乘以标准差的倒数(除以标准差)
    # 如此运算之后，mean=0，std=1
    X_train_scaled = (X_train - mean_on_train) / std_on_train
    # 对测试集做相同的变换（使用训练集的平均值和标准差）
    X_test_scaled = (X_test - mean_on_train) / std_on_train
    ```

    ```python
    mlp = MLPClassifier(random_state=0)
    mlp.fit(X_train_scaled, y_train)
    ...
    ```

    ```python
    Accuracy on training set: 0.991
    Accuracy on test set: 0.965
    ConvergenceWarning:
        Stochastic Optimizer: Maximum iterations reached and the optimization hasn't 	 converged yet.
    ```

    模型给出了一个警告，告诉我们已经达到最大迭代次数,应该增加迭代次数：

    ```python
    mlp = MLPClassifier(max_iter=1000, random_state=0) # max_iter=1000
    mlp.fit(X_train_scaled, y_train)
    ...
    ```

    ```python
    Accuracy on training set: 0.995
    Accuracy on test set: 0.965
    ```

    增加迭代次数仅**提高了训练集性能**，但**没有提高泛化性能**

    由于训练性能和测试性能之间仍有一些差距，所以我们可以尝试**降低模型复杂度**来得到更**好**

    **的泛化性能**。这里我们选择**增大alpha参数**（变化范围**相当大**，从**0.0001到1**），以此向

    **权重**添加**更强的正则化**：

    ```python
    mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
    mlp.fit(X_train_scaled, y_train)
    ...
    ```

    ```python
    Accuracy on training set: 0.988
    Accuracy on test set: 0.972
    ```

    这得到了与我们**目前最好**的模型相同的性能

    虽然可以分析神经网络学到了什么，但这通常比分析线性模型或基于树的模型**更为复杂**

    观察神经网络在乳腺癌数据集上学到的第一个隐层权重的**热图**(heat map):

    <img src=".\ML_COURSE\heat map neural network.png" style="zoom:50%;" />

  
  
#####   3.优点、缺点和参数

- 在机器学习的许多应用中，神经网络再次成为**最先进**的模型。它的**主要优点之一**是能够获

  取大量数据中包含的信息，并构建**无比复杂**的模型。给定**足够的计算时间和数据**，并且**仔**

  **细调节（tuning）参数**，神经网络**通常可以打败**其他机器学习算法（**无论**是**分类任务**还是**回归任务**）

- **缺点**:神经网络——特别是功能强大的大型神经网络——通常需要**很长的训练时间**,还需要仔细地**预处理**数据，

  神经网络**调参**本身也是一门艺术（Tuning neural network parameters is also an art onto itself）

- 神经网络调参的**常用方法**是，首先创建一个**大到足以过拟合**的网络，确保这个网络可以对任务进行学习。知道训练数据可以被学习之后，要么**缩小(shrink)网络**，要么**增大alpha**来**增强正则化**，这可以**提高泛化性能**

- **solver参数**设定：

  - 默认选项是**'adam'**，对数据的**缩放**相当**敏感**，始终将数据缩放为均值为0、方差为1（unit variance）是很重要的
  - 另一个选项是**'l-bfgs'**，其**鲁棒性**相当好，但在大型模型或大型数据集上的**时间会比较长**
  - 更高级的**'sgd'**选项，许多深度学习研究人员都会用到
  - 使用MLP时，建议使用'adam'和'l-bfgs'

### 2.4 分类器的不确定度估计 Uncertainty Estimates from Classifiers

scikit-learn接口的另一个有用之处，就是分类器能够给出预测的**不确定度估计**,不仅是分类器会预测一个测试点属于哪个类别，还包括它对这个预测的**置信程度**,scikit-learn中有两个函数可用于获取分类器的不确定度估计：**decision_function**（决策函数）和**predict_proba**（预测概率）

- 构建一个GradientBoostingClassifier分类器（同时拥有decision_function和predict_proba两个方法）

//TODO...

#### 2.4.1 决策函数 Decision Function

#### 2.4.2 预测概率 Predicting Probabilities

- predict_proba的输出是每个类别的概率，通常比decision_function的输出更容易理解。

- **过拟合更强**的模型可能会做出**置信程度更高**的预测，**即使**可能是**错**的。**复杂度越低**的模型通常对预测的**不确定度越大**。如果模型给出的不确定度**符合实际情况**，那么这个模型被称为**校正（calibrated）模型**。在校正模型

  中，如果预测有70%的确定度，那么它在70%的情况下正确

#### 2.4.3 多分类问题的不确定度 Uncertainty in Multiclass Classification

- 将这两个函数应用于**鸢尾花（Iris）数据集**，这是一个**三分类**数据集
- //TODO...

### 2.5 小结与展望 Summary and Outlook

关于何时使用哪种模型，下面是一份**快速总结**:

#### 快速总结 （由简单到复杂的模型）

##### #1 最近邻 Nearest neighbors

​	适用于**小型数据集**，是很好的基准（baseline）模型，很容易解释

##### #2 线性模型 Linear models

​	非常可靠的**首选算法**，适用于**非常大的数据集**，也适用于**高维数据**。

##### #3 朴素贝叶斯 Naive Bayes

​	**只适用于分类问题**。比线性模型速度还快，适用于**非常大的数据集**和**高维数据**。**精度**通

​	常要**低于线性模型**（less accurate）

##### #4 决策树 Decision trees

​	速度很**快**，**不需要数据缩放**，可以**可视化**，很**容易解释**

##### #5 随机森林 Random forests

​	几乎总是**比单棵**决策树的表现要**好**，**鲁棒性很好**，非常**强大**。**不需要数据缩放**。**不适用**

​	**于高维（high-dimensional）稀疏（sparse）数据**

##### #6 梯度提升决策树 Gradient boosted decision trees

​	**精度**通常**比随机森林略高**。与随机森林相比，**训练**速度**更慢**，但**预测**速度**更快**，需要的

​	**内存也更少**。比随机森林需要**更多的参数调节**

##### #7 支持向量机 Support vector machines

​	对于**特征含义相似**的**中等**大小的**数据集**很强大。**需要数据缩放**，**对参数敏感**

##### #8 神经网络 Neural networks

​	可以构建**非常复杂**的模型，**特别**是对于**大型数据集**而言。对**数据缩放敏感**，对**参数选取敏感**。

​	大型网络需要**很长的训练时间**

## Chapter3 Unsupervised Learning and Preprocessing 无监督学习与预处理

### 3.1 无监督学习的类型 Types of unsupervised learning

本章将研究**两种**类型的无监督学习：**数据集变换**与**聚类**

- **无监督变换（unsupervised transformation）**
  - 常见**应用**：**降维（dimensionality reduction）**
  - 另一个**应用**是找到“构成”数据的各个组成部分（例子：对文本文档集合进行主题提取）
- **聚类算法（clustering algorithm）**
  - 将数据划分成**不同的组**，每组包含**相似的物项**

### 3.2 无监督学习的挑战 Challenges in unsupervised learning

- **主要挑战**就是评估算法是否学到了有用的东西，无监督学习算法一般用于**不包含任何标签**信息的数据，所以**不知道**正确的**输出**应该是什么。因此很难判断一个模型是否“表现很好”。
- 通常来说，**评估无监督算法结果**的**唯一**方法就是**人工检查**（manually）
- 无监督算法通常可用于**探索性**的目的，而不是作为大型自动化系统的一部分
- 无监督算法的另一个**常见应用**是**作为监督算法**的**预处理步骤**

### 3.3 预处理与缩放 Preprocessing and Scaling

#### 3.3.1 不同类型的预处理 Diferent kinds of preprocessing

- StandardScaler：确保每个特征的**平均值（mean）为0**、**方差(variance)为1**，使所有特征都位于同一量

  级，但**不能保证**特征任何特定的**最大值**和**最小值**

- RobustScaler：工作原理与StandardScaler类似，使用**中位数**和**四分位数**。这样RobustScaler会忽略与其他点有很大不同的数据点（比如**测量误差**）。这些与众不同的数据点也叫**异常值（outlier）**

- MinMaxScaler：**移动**数据，使所有特征都刚好位于0到1之间

- Normalizer：用到一种完全不同的缩放方法，使得特征向量的**欧式长度等于1**。每个数据点的缩放比例都**不相同**（乘以其长度的倒数）。如果只有**数据的方向（或角度）**是**重要**的，而**特征向量的长度无关紧要**，那么通常会使用这种**归一化**（据点投射到半径为**1**的圆上（对于更高维度的情况，是球面））。

- **对比图**

  <img src=".\ML_COURSE\different Scaler.png" style="zoom: 50%;" />

#### 3.3.2 应用数据变换 Applying data transformations

- 将核**SVM**（SVC）应用在**cancer**数据集上，并使用**MinMaxScaler**来预处理数据

  ```python
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler() # MinMaxScaler
  scaler.fit(X_train) #X_train
  MinMaxScaler(copy=True, feature_range=(0, 1))
  ```

  ```python
  # 变换数据
  X_train_scaled = scaler.transform(X_train) #transform
  # 在缩放之前和之后分别打印数据集属性
  print("transformed shape: {}".format(X_train_scaled.shape))
  print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
  print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
  print("per-feature minimum after scaling:\n {}".format(    X_train_scaled.min(axis=0)))
  print("per-feature maximum after scaling:\n {}".format(    X_train_scaled.max(axis=0)))
  ```

  对**测试集**进行变换:

  ```python
  # 对测试数据进行变换
  X_test_scaled = scaler.transform(X_test)
  # 在缩放之后打印测试数据的属性
  ...min
  ...max
  ```

  <img src=".\ML_COURSE\negative_scaled.png" style="zoom:50%;" />

  对测试集缩放后的最大值和最小值不是1和0，因为MinMaxScaler（以及其他所有缩放器）总是

  对训练集和测试集应用**完全相同的变换**。也就是说，**transform**方法总是减去**训练集**的最小值，然后除以**训练集**的范围，而这两个值可能与测试集的最小值和范围并**不相同**

#### 3.3.3 对训练数据和测试数据进行相同的缩放 Scaling training and test data the same way

对训练集和测试集**分别**缩放：

```python
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)
```

**所有具有transform**方法的模型也都具有一个**fit_transform**方法：

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # StandardScaler
# 依次调用fit和transform（使用方法链）
X_scaled = scaler.fit(X).transform(X)
# 结果相同，但计算更加高效
X_scaled_d = scaler.fit_transform(X)
```

#### 3.3.4 预处理对监督学习的作用 The efect of preprocessing on supervised learning

- **观察使用MinMaxScaler对学习SVC的作用（缩放前后对比）**

- ```python
  from sklearn.svm import SVC
  X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
   random_state=0)
  svm = SVC(C=100)
  svm.fit(X_train, y_train)
  print(svm.score(X_test, y_test))
  ```

  ```python
  0.629370629371
  ```

- ```python
  # preprocessing using 0-1 scaling 使用0-1缩放进行预处理
  scaler = MinMaxScaler()
  scaler.fit(X_train) #X_train
  X_train_scaled = scaler.transform(X_train)
  X_test_scaled = scaler.transform(X_test) # 使用训练集transform测试集
  # learning an SVM on the scaled training data 在缩放数据上学习SVM
  svm.fit(X_train_scaled, y_train)
  # scoring on the scaled test set 在缩放的测试集上计算分数
  svm.score(X_test_scaled, y_test)
  ```
  
```python
  0.965034965034965
```

- ```python
  # preprocessing using zero mean and unit variance scaling 利用零均值和单位方差的缩放方法进行预处理
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  scaler.fit(X_train)
  X_train_scaled = scaler.transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  # learning an SVM on the scaled training data
  svm.fit(X_train_scaled, y_train)
  # scoring on the scaled test set
  svm.score(X_test_scaled, y_test)
  ```

  ```python
  0.95804195804195802
  ```

### 3.4　降维、特征提取与流形学习 Dimensionality Reduction, Feature Extraction and Manifold Learning 

利用无监督学习进行数据变换可能有很多种目的。最常见的目的就是可视化、压缩数据，以及寻找信息量更大的数据表示以用于进一步的处理。为了实现这些目的，最简单也最常用的一种算法就是**主成分分析**。以及另外两种算法：**非负矩阵分解（NMF）**和**t-SNE**，前者通常用于**特征提取**，后者通常用于**二维散点图的可视化**

#### 3.4.1　主成分分析 Principal Component Analysis (PCA)

主成分分析（PCA）是一种**旋转数据集**的方法，旋转后的特征在**统计上不相关**，在做完这种旋转之后，通常是根据**新特征**对**解释数据的重要性**来**选择**它的一个**子集**

- PCA对一个**模拟二维数据集**的作用：

  `mglearn.plots.plot_pca_illustration()`

  <img src=".\ML_COURSE\pca illustration.png" style="zoom:50%;" />

  - **第一张图**（左上）显示的是原始数据点，用不同颜色加以区分。算法首先找到**方差最大**的

    **方向**，将其标记为“**成分1**”（Component 1）。这是数据中**包含最多信息**的**方向（或向量）**，

    换句话说，沿着这个方向的**特征之间最为相关**

  - 然后，算法找到与第一个方向**正交**（成直角）且包**含最多信息**的方向。在**二维**空间中，只有**一个**成直角的方向，但在**更高维**的空间中会有**（无穷）多**的正交方向

  - 虽然这两个成分都画成箭头，但其头尾的位置并不重要

  - 利用这一过程找到的**方向**被称为**主成分**（principal component），因为它们是**数据方差的主要方向**。一般来说，**主成分的个数与原始特征相同**

  - **第二张图**旋转，使得第一主成分与x轴平行且第二主成分与y轴平行

  - 在旋转之前，从数据中**减去平均值**，使得变换后的数据**以零为中心**

  - 在PCA找到的旋转表示中，两个**坐标轴**是**不相关**的，也就是说，对于这种数据表示，

    **除了对角线，相关矩阵全部为零**

  - **第三张图**通过仅保留**一部分主成分**来使用PCA进行**降维**

  - 最后，我们可以**反向旋转**（undo the rotation）并将**平均值重新加到数据中**，得到**最后一张图**中的数据，这些数据中仅保留了**第一主成分**中包含的信息。

    这种变换有时用于**去除**数据中的**噪声**影响，或者将**主成分**中保留的那部分信息**可视化**

##### 1.将PCA应用于cancer数据集并可视化

PCA最常见的应用之一就是将**高维数据集可视化**，对于有两个以上特征的数据，很难绘制散点图，所以可以使用一种更简单的可视化方法——对**每个特征**分别**计算两个类别**（良性肿瘤和恶性肿瘤）的**直方图**

<img src=".\ML_COURSE\pca cancer.png" style="zoom:60%;" />

- 为每个特征创建一个直方图，计算具有某一特征的数据点在**特定范围内**（叫作**bin**）的**出现频率**。每张图都包含两个直方图，一个是良性类别的所有点（蓝色），一个是恶性类别的所有点（红色）。这样我们可以了解每个特征在**两个类别中**的**分布情况**，也可以猜测**哪些特征**能够**更好**地**区分**良性样本和恶性样本

- 利用PCA，我们可以获取到主要的**相互作用**，并得到稍为完整的图像。我们可以找到前两个主成分，并在这个新的二维空间中用**散点图**将数据可视化

  在应用PCA之前，利用StandardScaler缩放数据，使每个特征的方差均为1:

  ```python
  # StandardScaler
  scaler = StandardScaler()
  scaler.fit(cancer.data)
  X_scaled = scaler.transform(cancer.data)
  
  from sklearn.decomposition import PCA 
  # 保留数据的前两个主成分
  pca = PCA(n_components=2)
  # 对乳腺癌数据拟合PCA模型
  pca.fit(X_scaled)
  
  # 将数据变换到前两个主成分的方向上
  X_pca = pca.transform(X_scaled) # pca.transform(X_scaled)
  print("Original shape: {}".format(str(X_scaled.shape)))
  print("Reduced shape: {}".format(str(X_pca.shape)))
  ```

  ```python
  Original shape: (569, 30)
  Reduced shape: (569, 2)
  ```

  对前两个主成分作图:

  ```python
  ...
  mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
  ...
  ```

  ​												利用**前两个主成分**绘制乳腺癌数据集的二维散点图:

  <img src=".\ML_COURSE\pca cancer scatter.png" style="zoom:50%;" />

  - 重要的是要注意，PCA是一种**无监督**方法，在寻找旋转方向时**没有用到任何类别信息**。它

    只是**观察**数据中的**相关性**

  - PCA的一个**缺点**（downside）在于，通常不容易对图中的两个轴做出解释（interpret）

  - 在拟合过程中，主成分被保存在PCA对象的**components_属性**中：

    ```python
    print("PCA component shape: {}".format(pca.components_.shape))
    ```

    ```python
    PCA component shape: (2, 30)
    ```

  - components_中的每一**行**对应于一个**主成分**，它们按重要性排序（第一主成分排在首位，以此类推），**列**对应于PCA的**原始特征属性**

  - 可以用**热图**（heatmap）将系数可视化：

    <img src=".\ML_COURSE\pca cancer heatmap.png" style="zoom: 67%;" />

    - 在第一个主成分中，所有特征的符号相同（均为**正**，但前面提到过，箭头指向哪个方向无关紧要）,

      这意味着在所有特征之间存在普遍的**相关性**（如果一个测量值较大的话，其他的测量值可能也较大）

    - 第二个主成分的符号**有正有负**

    - 而且两个主成分都包含**所有30个特征**

##### 2.特征提取的特征脸 Eigenfaces for feature extraction

PCA的另一个应用是**特征提取**，特征提取背后的思想是，可以**找到一种数据表示**，比给定的原始表示**更适合于分析**。它的一个很好的应用实例就是**图像**

- 给出用PCA对图像做特征提取的一个简单应用，即处理**Wild数据集Labeled Faces**（标记人脸）中的人脸图像

- //TODO...见书

- 对PCA变换的介绍是：**先旋转数据，然后删除方差较小的成分**

- **另一种**有用的解释是尝试找到一些**数字**（**PCA旋转后的新特征值**），使我们可以将测试点表示为**主成分**的**加权求和**

- 图解PCA(Schematic view of PCA)：将图像分解(decomposing)为成分的加权求和:

  <img src=".\ML_COURSE\face decomposition.png" style="zoom:50%;" />

  这里x0、x1等是这个数据点的**主成分的系数**，换句话说，它们是图像在**旋转后**的**空间中的表示**(representation of the image in the rotated space)

- 另一种方法来理解PCA模型，就是仅使用一些成分对原始数据进行**重建**

  - 反向旋转并重新加上平均值

  - 可以对人脸做类似的变换，将数据降维到只包含一些主成分，然后反向旋转回到原始空间。回到

    **原始特征空间**可以通过inverse_transform方法来实现

- 随着使用的**主成分越来越多**，图像中也保留（preserved）了越来越多的**细节**，相当于把图分解式的求和中包含**越来越多的项**（extending the sum in Figure decomposition to include more and more terms）

- 与cancer类似，尝试使用PCA的**前两个主成分**，将数据集中的所有人脸在**散点图**中可视化

  ```python
  # mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
  mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
  ```

  <img src=".\ML_COURSE\face scatter plot pca.png" style="zoom:50%;" />

  如果只使用前两个主成分，整个数据只是一大团，看不到类别之间的分界，说明PCA只能捕捉到人脸非常粗略的特征

#### 3.4.2　非负矩阵分解 Non-Negative Matrix Factorization (NMF)

非负矩阵分解（non-negative matrix factorization，NMF）是另一种无监督学习算法，其目的在于**提取有用的特征**。它的**工作原理类似于PCA**，**也可以**用于**降维**。**与PCA相同**，试图将每个数据点写成一些**分量的加权求和**

- 在**PCA**中，想要的是**正交分量**，并且能够解释**尽可能多**的**数据方差**

- 而在**NMF**中，希望**分量和系数均为非负**，也就是说，我们希望分量和系数都大于或等于0。因此，这种方法**只能**应用于**每个特征**都是**非负**的数据，因为非负分量的非负求和**不可能**变为负值

- 将数据分解成非负加权求和的这个过程，对由**多个独立源**相加（或叠加）创建而成的数据特别有用

  （比如：识别出组成合成数据（多人说话的音轨或包含多种乐器的音乐）的**原始分量**）

- 与PCA相比，NMF得到的分量**更容易解释**，因为**负**的分量和系数可能会导致难以解释的**抵消效应**（cancellation effect）

##### 1.将NMF应用于模拟数据

- NMF能够**对数据进行操作**。这说明数据相对于原点**(0, 0)**的位置实际上对NMF很重要

- 可以将提取出来的非负分量看作是从(0, 0)到数据的**方向**（类似于向量）

- **两个分量**的非负矩阵分解（左）和**一个分量**的非负矩阵分解（右）找到的分量：

  <img src=".\ML_COURSE\one &amp; two components NMF.png" style="zoom:50%;" />

- 所有数据点都可以写成这两个分量的正数组合。如果有**足够多**的分量能够**完美地重建**数据（**分量个数与特征个数相同**），那么算法会选择指向**数据极值**的方向
- 如果仅使用**一个分量**，那么NMF会创建一个**指向平均值**的分量
- **与PCA不同**，减少分量个数不仅会**删除一些方向**，而且会**创建**一组**完全不同的分量**
- NMF的分量也**没有**按任何特定方法**排序**，所以不存在“第一非负分量”：**所有分量**的**地位平等**
- NMF使用了**随机初始化**，根据随机种子的不同可能会产生不同的结果，在复杂情况下，随机影响可能会很大

##### 2.将NMF应用于人脸图像

NMF的**主要参数**是需要提取的**分量个数**，要**小于**输入特征的个数，否则针对每个像素点都成为了单独的分量

- 观察**分量个数**如何影响NMF**重建**（restruct）数据的好坏：反向变换的数据质量与使用PCA时类似，但要**稍差**一些。这是符合预期的，因为**PCA**找到的是**重建**的**最佳方向**(optimum directions)，**NMF**通常**并不用于**对数据进行重建或编码，而是用于在数据中寻找有趣的模式（rather for finding interesting patterns within the data）
- 由于**分量**都是**正**的，因此比PCA分量**更像**人脸原型
- 分量系数（coefficient）较大的人脸更加清晰

提取这样的模式**最适合于**具有**叠加结构**的数据，包括**音频、基因表达和文本数据**

- 假设有100台装置测量一个由三个不同信号源合成的信号：

  代码见书 P125 

- 使用NMF来还原这三个信号：

  ```python
  nmf = NMF(n_components=3, random_state=42)
  S_ = nmf.fit_transform(X)
  print("Recovered signal shape: {}".format(S_.shape))
  ```

  ```python
  Recovered signal shape: (2000, 3)
  ```

  为了**对比**，也应用了**PCA**：

  ```python
  pca = PCA(n_components=3)
  H = pca.fit_transform(X)
  ```

  ...

  ​															利用NMF和PCA**还原**混合信号源:

  <img src=".\ML_COURSE\recover mixed nmf pca.png" style="zoom:50%;" />

  NMF在发现原始信号源时得到了不错的结果，而PCA则失败了

  - 要记住，NMF生成的分量是**没有顺序**的，在这个例子中，NMF分量的顺序与原始信号完

    全相同（参见三条曲线的颜色），但这**纯属偶然**

#### 3.4.3　用t-SNE进行流形学习  Manifold learning with t-SNE

- 虽然PCA通常是用于变换数据的首选方法，使你能够用**散点图**将其可视化，但这一方法的性质（**先旋转然后减少方向**）**限制了其有效性**，有一类用于可视化的算法叫作**流形学习算法**（**manifold learning** algorithm），它允许进行更复杂的映射，通常也可以给出**更好的可视化**。其中特别有用的一个就是**t-SNE算法**

- 流形学习算法主要用于可视化，因此很少用来生成两个以上的新特征

- 其中的算法计算**训练数据**的一种**新表示**，但**不允许变换新数据**，这意味着这些算法**不能用于测试集**：更确切地说，它们**只能变换**用于**训练**的数据，流形学习对**探索性数据分析**是很有用的

- **t-SNE**背后的思想是找到数据的一个**二维表示**，尽可能地**保持**数据点之间的**距离**

- 对scikit-learn包含的一个**手写数字数据集**应用t-SNE流形学习算法：

  - 使用**PCA**对前两个主成分作图：

    ```python
    # 构建一个PCA模型
    pca = PCA(n_components=2)
    pca.fit(digits.data)
    # 将digits数据变换到前两个主成分的方向上
    digits_pca = pca.transform(digits.data)
    ...
    ```

    ...<img/>

    大部分其他数字都大量**重叠**在一起

  - 将**t-SNE**应用于同一个数据集，并对结果进行比较:

    由于t-SNE**不支持变换新数据**，所以TSNE类**没有transform**方法。我们可以调用**fit_transform**方法来代替（构建模型并立刻返回变换后的数据）

    ```python
    from sklearn.manifold import TSNE
    tsne = TSNE(random_state=42)
    # 使用fit_transform而不是fit，因为TSNE没有transform方法
    digits_tsne = tsne.fit_transform(digits.data)
    ...
    ```

    利用**t-SNE**找到的两个分量绘制**digits数据集**的**散点图**:

    <img src=".\ML_COURSE\t-sne digits scatter.png" style="zoom:50%;" />

    - 数字1和9被分成几块，但大多数类别都形成一个**密集的组**。

    - **要记住**，这种方法**并不知道类别标签**：它完全是**无监督**的

    - 但它能够找到数据的一种二维表示，仅根据原始空间中**数据点**之间的**靠近程度**就能够将各个类别明

      确分开

    - 可尝试修改perplexity和early_exaggeration的调节参数，但作用一般很小

### 3.5  聚类 Clustering

聚类（clustering）是将数据集划分成组的任务，这些组叫作簇（cluster）,其目标是划分数据，使得**一个簇内的数据点非常相似且不同簇内的数据点非常不同**，聚类算法为每个数据点分配（或预测）一个**数字**，表示这个点属于哪个簇

#### 3.5.1 k均值聚类 k-Means clustering 

- 试图找到代表数据特定区域的**簇中心**（cluster center）

- **算法步骤**（迭代）：将每个数据点分配给**最近**的**簇中心**，然后将每个**簇中心设置为**所分配的所有数据点的**平均值**

- 将KMeans应用于模拟数据集make_blobs:

  ```python
  from sklearn.datasets import make_blobs 
  from sklearn.cluster import KMeans
  # 生成模拟的二维数据
  X, y = make_blobs(random_state=1)
  # 构建聚类模型
  kmeans = KMeans(n_clusters=3) # 寻找的簇个数为3
  kmeans.fit(X)
  ```

- 算法运行期间，为X中的每个训练数据点分配一个簇标签（label）

  ```python
  print("Cluster memberships:\n{}".format(kmeans.labels_))
  ```

  ```python
  Cluster memberships:
  [1 2 2 2 0 0 0 2 1 1 2 2 0 1 0 0 0 1 2 2 0 2 0 1 2 0 0 1 1 0 1 1 0 1 2 0 2 2 2 0 0 2 1 2 2 0 1 1 1 1 2 0 0 0 1 0 2 2 1 1 2 0 0 2 2 0 1 0 1 2 2 2 0 1  1 2 0 0 1 2 1 2 2 0 1 1 1 1 2 1 0 1 1 2 2 0 0 1 0 1]
  ```

- 可以用**predict**方法为**新数据点分配簇标签**。预测时会将最近的簇中心分配给每个新数据点，但现有模型不会改变。对训练集运行predict会返回与labels_**相同**的结果

  ```python
  print(kmeans.predict(X))
  ```

  ```python
  Cluster memberships:
  ...
  ```

##### 1.k均值的失败案例

- 每个簇都是凸形（convex）,因此，k均值只能找到**相对简单**的**形状**

- k均值还假设**所有方向**对每个簇都**同等重要**

  例如：(k均值无法识别非球形簇以及复杂形状(two_moons)的簇)

  <img src=".\ML_COURSE\kmeans failed.png" style="zoom:50%;" />

<img src=".\ML_COURSE\kmeans failed in complex.png" style="zoom:50%;" />

##### 2.矢量量化，或者将k均值看作分解

- 虽然k均值是一种聚类算法，但在k均值和分解方法（比如之前讨论过的PCA和NMF）之间存在一些有趣的**相似**之处

- **PCA**试图找到数据中**方差最大的方向**，而**NMF**试图找到**累加的分量**，这通常对应于数据的“极值”或“部分”。两种方法都试图将数据点表示为一些**分量之和**

- 与之相反，k均值则尝试利用**簇中心**来表示每个数据点（可以将其看作**仅用一个分量**来表示每个数据点，该分量由簇中心给出）

- 这种观点将k均值看作是一种**分解方法**，其中**每个点用单一分量来表示**，这种观点被称为**矢量量化（vector quantization）**

- 并排**比较**PCA、NMF和k均值：

  ```python
  X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
  nmf = NMF(n_components=100, random_state=0) # NMF
  nmf.fit(X_train)
  pca = PCA(n_components=100, random_state=0) # PCA
  pca.fit(X_train)
  kmeans = KMeans(n_clusters=100, random_state=0) # KMeans
  kmeans.fit(X_train)
  ```

  **重建（reconstructed）:**

  ```python
  X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
  X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]
  X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)
  ```

- //TODO...

- 利用k均值的许多簇来表示复杂数据集中的变化

  <img src=".\ML_COURSE\many k-means clusters cover variation.png" style="zoom:50%;" />

- 将**到每个簇中心的距离**作为**特征**，还可以得到一种**表现力(expressive)更强**的数据表示。可以利用kmeans的**transform**方法来完成这一点

  ```python
  distance_features = kmeans.transform(X)
  print("Distance feature shape: {}".format(distance_features.shape))
  print("Distance features:\n{}".format(distance_features))
  ```

  ```python
  Distance feature shape: (200, 10)
  Distance features:
  ...
  ```

- k均值是非常流行的聚类算法，因为它不仅**相对容易理解和实现**，而且**运行速度**也**相对较**

  **快**。k均值可以轻松扩展到**大型数据集**，scikit-learn甚至在MiniBatchKMeans类中包含了

  一种更具可扩展性的变体，可以处理非常大的数据集

  **缺点：**

- k均值的**缺点**(drawback)之一在于，它依赖于随机初始化，也就是说，算法的输出依赖于**随机种子**。

  **默认**情况下，scikit-learn用**10种**不同的随机初始化将算法运行**10次**，并返回最佳结果

- k均值还有一个**缺点**，就是对**簇形状**的假设的**约束性较强**，而且还要求**指定**所要寻找的**簇的个数**（在现实世界的应用中可能并不知道这个数字）

#### 3.5.2 凝聚聚类 Agglomerative Clustering 

- 凝聚聚类（agglomerative clustering）指的是许多基于**相同原则**构建的聚类算法，这一原则是：算法首先声明每个点是**自己的簇**，然后**合并**两个**最相似**的簇，直到满足某种**停止准则**为止

- **scikit-learn**中实现的停止准则是**簇的个数**，因此相似的簇被合并，直到仅剩下指定个数的簇

- 还有一些**链接（linkage）准则**，规定**如何度量**“最相似的簇”。这种度量总是定义在两个现有的簇之间

- scikit-learn中实现了以下三种选项：

  - **ward**

    **默认**选项。ward挑选两个簇来合并，使得所有簇中的**方差增加最小**。这通常会得到**大小差不多相等**的簇

  - **average**

    average链接将簇中所有点之间**平均距离最小**的两个簇合并

  - **complete**

    complete链接（也称为最大链接）将簇**中点之间最大距离最小**的两个簇合并

  - ward适用于**大多数**数据集，如果簇中的成员个数**非常不同**（比如其中一个比其他所有都**大得多**），那么average或complete可能效果**更好**

- //TODO 寻找三个簇（二维数据集+简单三簇）...

- 由于算法的工作原理，凝聚算法**不能对新数据点**做出**预测**。因此AgglomerativeClustering**没有predict**方

  法。为了构造模型并得到**训练集上簇的成员关系**，可以改用**fit_predict**方法

  ```python
  from sklearn.cluster import AgglomerativeClustering
  X, y = make_blobs(random_state=1)
  agg = AgglomerativeClustering(n_clusters=3) # n_clusters=3
  assignment = agg.fit_predict(X)
  ...
  ```

  虽然凝聚聚类的scikit-learn实现需要你指定希望算法找到的**簇的个数**，但凝聚聚类方法为**选择正确的个数**提供了一些帮助，将在下面讨论

##### 1.层次聚类与树状图 Hierarchical clustering and dendrograms

- 凝聚聚类生成了所谓的层次聚类（hierarchical clustering）。聚类过程**迭代**进行，每个点都从一个单点簇变为属于最终的某个簇，但它依赖于数据的二维性质，因此不能用于具有两个以上特征的数据集

- 另一个将层次聚类**可视化的工具**，叫作**树状图**（dendrogram），它可以处理多维数据集

- 目前scikit-learn没有绘制树状图的功能。但你可以利用**SciPy**轻松生成树状图

- SciPy提供了一个函数，接受数据**数组X**并计算出一个**链接数组**（linkage array），它对层次聚类的**相似度**进行**编码**。然后我们可以将这个链接数组提供给**scipy**的**dendrogram函数**来绘制树状图：

  ```python
  # 从SciPy中导入dendrogram函数和ward聚类函数
  from scipy.cluster.hierarchy import dendrogram, ward
  X, y = make_blobs(random_state=0, n_samples=12)
  # 将ward聚类应用于数据数组X
  # SciPy的ward函数返回一个数组，指定执行凝聚聚类时跨越的距离
  linkage_array = ward(X)
  # 现在为包含簇之间距离的linkage_array绘制树状图
  dendrogram(linkage_array)
  
  # 在树中标记划分成两个簇或三个簇的位置
  ax = plt.gca()
  bounds = ax.get_xbound()
  ax.plot(bounds, [7.25, 7.25], '--', c='k')
  ax.plot(bounds, [4, 4], '--', c='k')
  ...
  ```

  <img src=".\ML_COURSE\three clusters dendrogram.png" style="zoom:50%;" />

  - 树状图在底部显示**数据点**（编号从0到11）。然后以这些点（表示**单点簇**）作为**叶节点**绘

    制一棵树，每**合并两个簇**就添加一个**新的父节点**

  - 从下往上看，数据点1和4首先被合并，以此类推...

  - 在这张树状图中，**最长的分支**是用标记为“three clusters”（三个簇）的虚线表示的三条线

  - 不幸的是，凝聚聚类仍然**无法**分离像two_moons数据集这样复杂的形状，下一个算法DBSCAN可以解决这个问题

#### 3.5.3 DBSCAN density-based spatial clustering of applications with noise

**具有噪声的基于密度的空间聚类应用**

- DBSCAN的**主要优点**是它**不需要**用户先验地**设置簇的个数**，可以划分具有**复杂形状**的簇，还可以**找出不属于**任何簇的**点**。DBSCAN比凝聚聚类和k均值**稍慢**，但仍可以扩展到**相对较大的数据集**

- DBSCAN的原理是识别特征空间的密集（dense）区域。DBSCAN背后的思想是，簇形成数据的密集区域，并由相对较空的区域分隔开

- 在密集区域内的点被称为**核心样本**（**core sample**，或核心点），它们的定义如下

- DBSCAN 有两个参数：**min_samples**和**eps**。如果在距一个给定数据点**eps的距离内至少有min_samples个数据点**，那么这个数据点就是**核心样本**。DBSCAN将**彼此距离小于eps**的**核心样本**放到**同一个簇**中

- 算法首先任意选取一个点，然后找到到这个点的距离小于等于eps的所有的点。如果距起始点的距离在eps之内的数据点个数**小于min_samples**，那么这个点被标记为**噪声（noise）**，也就是说它**不属于任何簇**

- 如果距离在eps之内的数据点个数**大于min_samples**（至少有min_samples个），则这个点被标记为核心样本，并被分配一个新的簇标签。然后访问该点的所有**邻居**（在距离eps以内），依照之前的条件以此类推分配标签，簇逐渐增大，直到在簇的eps距离内没有更多的核心样本为止

  算法迭代示意图：

  <img src=".\ML_COURSE\dbscan iterations.png" style="zoom:50%;" />

- 最后，一共有三种类型的点：**核心点**、与核心点的距离在eps之内的点（叫作**边界点**，boundary point）和**噪声**

- 边界点可能与**不止一个簇的核心样本相邻**。因此，边界点所属的簇依赖于数据点的访问顺序。一般来说只有很少的边界点，这种对访问顺序的轻度依赖并不重要

- 与凝聚聚类类似，DBSCAN也**不允许**对**新的测试数据**进行**预测**，所以将使用**fit_predict**方法来执行聚类并返回簇标签

  将DBSCAN应用于**演示凝聚聚类的模拟数据集**：

  ```python
  from sklearn.cluster import DBSCAN
  X, y = make_blobs(random_state=0, n_samples=12)
  dbscan = DBSCAN()
  clusters = dbscan.fit_predict(X)
  print("Cluster memberships:\n{}".format(clusters))
  ```

  ```python
  Cluster memberships:
  [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
  ```

  所有数据点都被分配了标签-1，这代表**噪声**，这是eps和min_samples默认参数设置的结果(**not tuned** for small toy datasets)

- min_samples和eps取不同值时的簇分类如下所示，及其可视化结果见图

  <img src=".\ML_COURSE\min samples eps.png" style="zoom:50%;" />

  <img src=".\ML_COURSE\min samples eps illustration.png" style="zoom: 67%;" />

  在这张图中，属于簇的点是实心的，而噪声点则显示为空心的。核心样本显示为较大的标

  记，而边界点则显示为较小的标记

- 将**eps**设置得**非常小**，意味着没有点是核心样本，可能会导致所有点都被标记为**噪声**。将eps设置得

  **非常大**，可能会导致所有点形成**单个簇**

- 设置**min_samples**主要是为了判断稀疏区域内的点被标记为异常值还是形成自己的簇，如

  果**增大min_samples**，任何一个包含少于min_samples个样本的簇现在将被标记为**噪声**。因

  此，**min_samples决定簇的最小尺寸**

- 虽然DBSCAN不需要显式地设置簇的个数，但设置eps可以**隐式**地控制找到的**簇的个数**

- 使用StandardScaler或MinMaxScaler对数据进行**缩放**之后，有时会更容易找到eps的较好

  取值，因为能确保所有特征具有相似的**范围**

- 在**two_moons数据集**上运行DBSCAN的结果，利用默认设置：

  ```python
  X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
  
  # 将数据缩放成平均值为0、方差为1 
  scaler = StandardScaler()scaler.fit(X)
  X_scaled = scaler.transform(X)
  
  dbscan = DBSCAN()
  clusters = dbscan.fit_predict(X_scaled) # fit_predict
  # 绘制簇分配
  plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
  plt.xlabel("Feature 0")
  plt.ylabel("Feature 1")
  ```
  
  ​													利用默认值eps=0.5的DBSCAN找到的簇分配
  
  <img src=".\ML_COURSE\eps two_moons.png" style="zoom:50%;" />
  
  

#### 3.5.4 聚类算法的对比与评估

##### 1.用真实值评估聚类 Evaluating clustering with ground truth

- **调整rand指数** （adjusted rand index，**ARI**）和**归一化互信息**（normalized mutual information，**NMI**）

- 使用ARI来比较k均值、凝聚聚类和DBSCAN算法

- ```python
  from sklearn.metrics.cluster import adjusted_rand_score # adjusted_rand_score
  ...
  ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__,                                           adjusted_rand_score(y, clusters)))
  ```
  
  ​				利用监督ARI分数在two_moons数据集上比较**随机分配**、**k均值**、**凝聚聚类**和**DBSCAN**
  
  <img src=".\ML_COURSE\ARI compare.png" style="zoom: 80%;" />
  
  用这种方式评估聚类时，一个常见的**错误**是使用accuracy_score而不是adjusted_rand_score、normalized_mutual_info_score或其他聚类指标

##### 2.在没有真实值的情况下评估聚类 Evaluating clustering without ground truth

- **轮廓系数（silhouette coeffcient）**轮廓分数计算一个簇的紧致度，其值越大越好，最高分数为1。虽然 

  紧致的簇很好，但紧致度不允许复杂的形状

- 利用轮廓分数在two_moons数据集上比较k均值、凝聚聚类和DBSCAN：

  ```python
  from sklearn.metrics.cluster import silhouette_score
  ...
  silhouette_score(X_scaled, clusters)
  ```

  <img src=".\ML_COURSE\silhouette_score.png" style="zoom:80%;" />

  k均值的轮廓分数最高，尽管我们可能更喜欢DBSCAN的结果。对于评估聚类，稍好的策略是使用**基于鲁棒性**的（robustness-based）聚类指标

#### 3.5.5 聚类方法小结  Summary of Clustering Methods

k均值和凝聚聚类允许你指定想要的簇的数量，而DBSCAN允许你用eps参数定义接近程度，从而间接影响簇的大小。三种方法都可以用于**大型**的**现实世界**数据集，都相对容易理解，也都可以聚类成多个簇

#### 估计器接口小结

<img src=".\ML_COURSE\估计器接口小结1.png" style="zoom: 67%;" />

<img src=".\ML_COURSE\估计器接口小结2.png" style="zoom:67%;" />

#### Test1 课堂练习

```python
1.将knn算法运用于iris数据集，比较MinMaxScaler前后的score变化
2.将KMeans算法运用于iris数据集(n_clusters=3),利用样本真实类别评价聚类结果

/ML/test1/class_test1.ipynb

#1 
#(1)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Test set accuracy: {:.2f}".format(knn.score(X_test, y_test)))

#(2)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)# MinMax缩放
X_test_scaled = scaler.transform(X_test)
# svm.fit(X_train_scaled, y_train)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_scaled, y_train)
print("Scaled test set accuracy: {:.2f}".format(knn.score(X_test_scaled, y_test)))
# X_train scaled
print("per-feature minimum before scaling:\n{}".format(X_train.min(axis=0))) 
print("per-feature maximum before scaling:\n{}".format(X_train.max(axis=0)))
print("per-feature minimum after scaling:\n{}".format(X_train_scaled.min(axis=0))) 
print("per-feature maximum after scaling:\n{}".format(X_train_scaled.max(axis=0)))
# X_test scaled (range changed not in [0,1])
print("per-feature minimum before scaling:\n{}".format(X_test.min(axis=0))) 
print("per-feature maximum before scaling:\n{}".format(X_test.max(axis=0)))
print("per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0))) 
print("per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))

#2
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris=load_iris()
data = iris.get('data')

kmeans = KMeans(n_clusters=3) 
kmeans.fit(data)

label_pred=kmeans.labels_
print(label_pred)
#print(kmeans.predict(data))

a=label_pred.tolist()
b=a[:50]
c=a[50:100]
d=a[100:150]
b1=max(max(b.count(0),b.count(1)),b.count(2))
print(b1) # b
c1=max(max(c.count(0),c.count(1)),c.count(2))
print(c1) # c
d1=max(max(d.count(0),d.count(1)),d.count(2))
print(d1) # d
print("accuracy: {:.2f}".format((b1+c1+d1)/150))
```



#### Test2 课堂练习

```python
1.使用凝聚聚类算法对鸢尾花数据集进行聚类，并画出对应的树状图。
2.使用DBSCAN算法对原始鸢尾花数据集进行聚类，查看聚类结果。将鸢尾花数据集进行最小最大缩放后，再次使用DBSCAN算法聚类，并查看聚类结果。
3.使用凝聚聚类算法对鸢尾花数据集进行聚类，分别使用adjusted_rand_score和silhouette_score指标对聚类结果进行评价。

/ML/test2/class_test2.ipynb

#1
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,ward
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris=load_iris()
data=iris.data
target=iris.target

linkage_array = ward(data)
dendrogram(linkage_array)
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')
ax.text(bounds[1], 7.25, ' two clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")

#2
#(1)
from sklearn.cluster import DBSCAN
dbscan = DBSCAN()
clusters = dbscan.fit_predict(data)
print("Cluster memberships:\n{}".format(clusters))

#(2)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()
scaler.fit(data)
X_scaled=scaler.transform(data)
# dbscan = DBSCAN(eps=0.4, min_samples=5)
dbscan=DBSCAN()
clusters = dbscan.fit_predict(X_scaled)
print("Cluster memberships:\n{}".format(clusters))

#3
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import silhouette_score
agg = AgglomerativeClustering(linkage='average',n_clusters=3)#linkage='ward' 
assignment = agg.fit_predict(data)
print(adjusted_rand_score(assignment,target)) #
print(silhouette_score(data,assignment)) #
```



#### Test3 课堂练习

```python
import mglearn
from sklearn.datasets import load_iris
data = load_iris()
y = data.target
x = data.data

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced_x=pca.fit_transform(x)

import matplotlib.pyplot as plt
mglearn.discrete_scatter(reduced_x[:,0],reduced_x[:,1],y)
plt.show()


-----------------------------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)
print(tree.score(X_test, y_test))
print(tree.feature_importances_)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(X_train,y_train)
print(forest.score(X_test, y_test))
print(forest.feature_importances_)
```



## Chapter4 Representing Data and Engineering Features 数据表示和特征工程 

- 分类特征（categorical feature）

- 离散特征（discrete feature）

- 对于某个特定应用来说，如何找到最佳数据表示，这个问题被称为**特征工程**（feature  

  engineering）

- adult数据集

#### 4.1.1 One-Hot编码（虚拟变量）One-Hot-Encoding (Dummy variables) 

- one-hot 编码（one-hot-encoding）或 N 取一编码（one-out-of-N encoding），也叫虚拟变量（dummy variable）

- 新特征取值为 0 和 1

- pandas库

- ```python
  import pandas as pd
  from IPython.display import display
  # 文件中没有包含列名称的表头，因此我们传入header=None
  # 然后在"names"中显式地提供列名称
  data = pd.read_csv(
   "data/adult.data", header=None, index_col=False,
   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
   'marital-status', 'occupation', 'relationship', 'race', 'gender',
   'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
   'income'])
  # 为了便于说明，我们只选了其中几列
  data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
   'occupation', 'income']]
  # IPython.display可以在Jupyter notebook中输出漂亮的格式
  display(data.head())
  ```

##### 1.**检查字符串编码的分类数据  Checking string-encoded categorical data** 

  ```python
  print(data.gender.value_counts())
  ```

  - get_dummies函数(自动变换所有具有对象类型（比如字符串）的列或所有分类的列)

  ```python
  In[4]:
  print("Original features:\n", list(data.columns), "\n")
  data_dummies = pd.get_dummies(data)
  print("Features after get_dummies:\n", list(data_dummies.columns))
  Out[4]:
  Original features:
   ['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation',
   'income']
  Features after get_dummies:
  ['age', 'hours-per-week', 'workclass_ ?', 'workclass_ Federal-gov',
   'workclass_ Local-gov', 'workclass_ Never-worked', 'workclass_ Private',
    ...
   'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners',
   ...
   'occupation_ Tech-support', 'occupation_ Transport-moving',
   'income_ <=50K', 'income_ >50K']
  ```

  - **连续特征** age 和 hours-per-week **没有发生变化**
  
  - 使用 values 属性将 data_dummies 数据框（DataFrame）转换为 **NumPy 数组**， 
  
    然后在其上训练一个机器学习模型。在训练模型之前，注意要把**目标变量**（现在被编码为两个 income 列）从数据中分离出来。将输出变量或输出变量的一些导出属性包含在特征表示中，这是构建监督机器学习模型时一个非常常见的**错误**。
  
  - 提取包含特征的列，也就是从 age 到 occupation_ Transport-moving 
  
    的所有列。这一范围包含所有特征，但不包含目标：
  
    ```python
    features = data_dummies.ix[:, 'age':'occupation_ Transport-moving']
    # 提取NumPy数组
    X = features.values
    y = data_dummies['income_ >50K'].values
    print("X.shape: {} y.shape: {}".format(X.shape, y.shape))
    Out[6]:
    X.shape: (32561, 44) y.shape: (32561,)
    ```
  
    现在数据的表示方式可以被 **scikit-learn** 处理
  
    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    print("Test score: {:.2f}".format(logreg.score(X_test, y_test)))
    ```
  
    ```python
    Test score: 0.81
    ```


#### 4.1.2 数字可以编码分类变量

#### 4.2 分箱、离散化、线性模型与树 Binning, Discretization, Linear Models and Trees

- 决策树可以构建更为复杂的数据模型，但这强烈依赖于数据表示。有一种方法可以让线性模型在连续

  数据上变得更加强大，就是使用特征分箱（binning，也叫离散化，即discretization）将其划分为多个特征











