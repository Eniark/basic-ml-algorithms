# ml-library

1. Implement basic ML algorithms from scratch using Python;
2. Animate training;



| Date | Progress |
|--- | --- |
| `06/06/2023` | Mainly focused on Polynomial regressions and regularization techniques like Ridge, Lasso and ElasticNet. Had to grab a book to refresh my knowledge and learn something new. Encountered a weird behaviour with polynomial regression with different datasets: on simple non-linear data it's performing ok, but on other (I used linear, but it shouldn't be an issue as far as I know.) the gradient explodes and so does the loss. Also, I'm getting a better hang of OOP and it's usefullness (inheritance is a godly feature!). |

| `06/10/2023` | Some progress in Regression file. Worked mainly on regularization techniques: Lasso, Ridge and ElasticNet. Created a regression solver using normal equation. Moreover, lasso regularization to Normal equation was added. Though I'm not sure if it's correctly working yet. Created also a few learning rate schedulers as I had some trouble with Lasso: the algorithm wouldn't converge during last iterations and would bounce around. Luckily, the book I'm reading had a perfect advice to use a learning rate scheduler with Lasso. And that's exactly what I did and it solved the problem on non-convergence. Also, the schedulers helped with polynomial regression. I figured out the reason for weird behaviour of the class: I was doing too many normalizations. Removing (or changing to standardization!) helped. Now it works flawlessly. Next plans: finish Normal equation and start work on KNN. | 