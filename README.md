# ml-library

1. Implement basic ML algorithms from scratch using Python;
2. Animate training;



| Date | Progress |
|--- | --- |
| `06/06/2023` | Mainly focused on Polynomial regressions and regularization techniques like Ridge, Lasso and ElasticNet. Had to grab a book to refresh my knowledge and learn something new. Encountered a weird behaviour with polynomial regression with different datasets: on simple non-linear data it's performing ok, but on other (I used linear, but it shouldn't be an issue as far as I know.) the gradient explodes and so does the loss. Also, I'm getting a better hang of OOP and it's usefullness (inheritance is a godly feature!). |
| `06/10/2023` | Some progress in Regression file. Worked mainly on regularization techniques: Lasso, Ridge and ElasticNet. Created a regression solver using normal equation. Moreover, lasso regularization to Normal equation was added. Though I'm not sure if it's correctly working yet. Created also a few learning rate schedulers as I had some trouble with Lasso: the algorithm wouldn't converge during last iterations and would bounce around. Luckily, the book I'm reading had a perfect advice to use a learning rate scheduler with Lasso. And that's exactly what I did and it solved the problem on non-convergence. Also, the schedulers helped with polynomial regression. I figured out the reason for weird behaviour of the class: I was doing too many normalizations. Removing (or changing to standardization!) helped. Now it works flawlessly. Next plans: finish Normal equation and start work on KNN. | 
| `06/11/2023` | Added regularization possibility to Normal equation. Implemented brute-force algorithm for KNN. Created KNeighboursClassifier and -Regressor. Started work on KD-Tree for KNN. |
| `06/18/2023` | Added KD-tree implementation. Implemented tree construction and traversal to search for k-nearest neighbors. |
| `06/23/2023` | Added Logistic Regression. Had some trouble with backpropagation but eventually found the mistake. Started implementation of One-vs-Rest classifiers since Logistic regression is a binary classifier. |
| `06/24/2023` | Added multiclass classification strategies (One-vs-Rest, One-vs-One). Tested it using Logistic regression. Tested Logistic regression with Lasso and Ridge regularization. Fixed a bug in KD-Tree when searching for closest neighbors. Added more documentation to code. |
| `06/29/2023` | Finishes One-vs-One multiclass strategy. Modified the structure of the project. As the project grows more and more files appear which makes it hard to navigate for me (not to mention the users who just came into this repo to look around). I've started working on SVM and I'm having some trouble with implementing it using numpy - different rules apply when computing gradients. I'm thinking about using plain for-loops to do the job. They are slow, but more understandable for me in this case. |