# Computational Optimization

## Objectives

* **OA1:** Explain the fundamental principles of classical numerical optimization methods in the context of AI.
* **OA2:** Implement and adapt metaheuristics to solve complex problems in AI.
* **OA3:** Apply gradient descent methods in the training of deep neural networks, tuning hyperparameters and analyzing convergence plots.
* **OA4:** Select and justify the choice of appropriate cost functions for different AI tasks, evaluating their impact on model performance.
* **OA5:** Critically evaluate and compare the performance of the optimization methods used, justifying decisions and proposing improvements.
* **OA6:** Integrate and adapt the chosen optimization strategies, considering practical constraints (time, resources, scalability) and robustness criteria, to obtain solutions tailored to the problem's needs.

## Program

* **CP1:** Classical numerical methods, derivatives and gradients, unconstrained and constrained optimization, introduction to convergence analysis.
* **CP2:** Metaheuristics: Simulated Annealing, Tabu Search, Evolutionary Algorithms; applications to complex problems (model selection, hyperparameter tuning, global optima search).
* **CP3:** Gradient descent (SGD, Momentum, Adagrad, Adam); implementation in deep neural networks; analysis of convergence plots and stability of training and validation processes.
* **CP4:** Loss/cost functions in Deep Learning: impact on convergence, generalization ability, and model robustness.
* **CP5:** Comparison of methods, stopping criteria, scalability, result analysis, improvement proposals, and discussion of practical cases.

## Course Overview
This study plan is structured to sequentially cover the objectives (OA1-OA6) and programmatic content (CP1-CP5) outlined in your syllabus. It bridges foundational mathematical concepts with cutting-edge artificial intelligence implementations.

The plan is divided into a **10-week curriculum**, assuming 8-10 hours of study per week. It includes theoretical readings, practical programming exercises, and high-quality video lectures.

## Part I: Foundations of Mathematical Optimization
**Mapping:** CP1, OA1

### Week 1: Derivatives, Gradients, and Unconstrained Optimization
* **Concepts:** Functions of multiple variables, partial derivatives, the Gradient vector, the Hessian matrix, Taylor series approximations. Unconstrained optimization basics (local vs. global minima).
* **Readings:**
    * *Mathematics for Machine Learning* by Deisenroth, Faisal, and Ong (Chapters 5 & 7).
    * *Numerical Optimization* by Nocedal & Wright (Chapters 1 & 2).
* **Videos:**
    * <a href="https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr" target="_blank" rel="noopener noreferrer">3Blue1Brown - Essence of Calculus (Partial Derivatives & Gradients)</a>
    * <a href="https://www.khanacademy.org/math/multivariable-calculus" target="_blank" rel="noopener noreferrer">Khan Academy - Multivariable Calculus (Gradient & Directional Derivatives)</a>
* **Practical Task:** Implement a basic vanilla gradient descent algorithm from scratch in Python using NumPy to find the minimum of a simple 2D quadratic function (e.g., $f(x, y) = x^2 + y^2$).

### Week 2: Constrained Optimization and Convergence Analysis
* **Concepts:** Equality and inequality constraints, Lagrange Multipliers, Karush-Kuhn-Tucker (KKT) conditions. Introduction to convergence rates (linear, superlinear, quadratic).
* **Readings:**
    * *Convex Optimization* by Stephen Boyd and Lieven Vandenberghe (Chapter 5: Duality).
* **Videos:**
    * <a href="https://www.youtube.com/playlist?list=PL3940DD956CDF0622" target="_blank" rel="noopener noreferrer">Stanford Convex Optimization Course (Prof. Stephen Boyd) - Lectures 1 & 5</a>
* **Practical Task:** Solve a constrained optimization problem manually using Lagrange multipliers, then verify the result using `scipy.optimize.minimize`.

## Part II: Metaheuristics and Complex Search Spaces
**Mapping:** CP2, OA2

### Week 3: Single-State Metaheuristics
* **Concepts:** Escaping local optima, Hill Climbing, Simulated Annealing (temperature scheduling, acceptance probability), Tabu Search (short-term memory, tabu lists).
* **Readings:**
    * *Essentials of Metaheuristics* by Sean Luke (Free online PDF - Chapters 1 & 2).
* **Videos:**
    * <a href="https://www.youtube.com/watch?v=q6fDgR_XjLU" target="_blank" rel="noopener noreferrer">Simulated Annealing - MIT OpenCourseWare (AI Course)</a>
* **Practical Task:** Implement Simulated Annealing to solve the Traveling Salesperson Problem (TSP) for a 20-city dataset. Track and plot the cost function over iterations.

### Week 4: Population-Based Metaheuristics
* **Concepts:** Evolutionary Algorithms (Genetic Algorithms), mutation, crossover, selection strategies. Applying these to hyperparameter tuning and model architecture selection.
* **Readings:**
    * *Essentials of Metaheuristics* by Sean Luke (Chapter 3: Evolutionary Algorithms).
* **Videos:**
    * <a href="https://www.youtube.com/watch?v=MacVqujSXWE" target="_blank" rel="noopener noreferrer">Genetic Algorithms Explained - Computerphile</a>
* **Practical Task:** Use a Genetic Algorithm framework (like `DEAP` in Python) to optimize the hyperparameters (learning rate, number of layers, neurons) of a basic Scikit-Learn Multi-Layer Perceptron (MLP).

## Part III: Advanced Gradient Descent and Deep Learning Optimizers
**Mapping:** CP3, OA3

### Week 5: Stochastic Gradient Descent and Momentum
* **Concepts:** Batch vs. Mini-batch vs. Stochastic Gradient Descent. The issue of ravines and saddle points. Momentum and Nesterov Accelerated Gradient (NAG).
* **Readings:**
    * *Deep Learning* by Goodfellow, Bengio, and Courville (Chapter 8: Optimization for Training Deep Models, Sections 8.1 - 8.3).
    * <a href="https://ruder.io/optimizing-gradient-descent/" target="_blank" rel="noopener noreferrer">Sebastian Ruder's Blog: An overview of gradient descent optimization algorithms</a>. *Essential reading.*
* **Videos:**
    * <a href="https://www.youtube.com/watch?v=_JB0AO7QxSA" target="_blank" rel="noopener noreferrer">Stanford CS231n - Lecture 7: Training Neural Networks II (Optimizers)</a>
* **Practical Task:** Train a simple CNN on the MNIST dataset using pure SGD, then SGD with Momentum. Plot the training and validation loss curves on the same graph to compare convergence speed.

### Week 6: Adaptive Learning Rate Methods
* **Concepts:** Adagrad, RMSProp, Adam, AdamW. How adaptive algorithms adjust learning rates per parameter. Analyzing convergence plots and training stability.
* **Readings:**
    * *Deep Learning* (Chapter 8, Section 8.5).
    * Paper: *Adam: A Method for Stochastic Optimization* (Kingma & Ba, 2014).
* **Videos:**
    * <a href="https://www.coursera.org/learn/deep-neural-network" target="_blank" rel="noopener noreferrer">Andrew Ng - Deep Learning Specialization (Course 2: Improving Deep Neural Networks, Week 2)</a>
* **Practical Task:** Implement Adam from scratch in Python. Then, use PyTorch to train a ResNet model on CIFAR-10 comparing RMSProp and Adam, tracking metrics using TensorBoard to analyze stability.

## Part IV: Cost Functions and Model Evaluation
**Mapping:** CP4, OA4

### Week 7: Designing and Selecting Loss Functions
* **Concepts:** Maximum Likelihood Estimation (MLE) as the basis for loss functions. Mean Squared Error (Regression), Binary/Categorical Cross-Entropy (Classification). Hinge Loss (SVMs).
* **Readings:**
    * *Deep Learning* (Chapter 5: Machine Learning Basics, Section 5.5).
* **Videos:**
    * <a href="https://www.youtube.com/watch?v=6ArSys5qHAU" target="_blank" rel="noopener noreferrer">StatQuest - Cross Entropy, Clearly Explained</a>
* **Practical Task:** Write a report analyzing *why* Cross-Entropy is preferred over MSE for classification tasks involving sigmoid/softmax outputs (focusing on gradient saturation).

### Week 8: Advanced Loss Functions and Robustness
* **Concepts:** Contrastive Loss, Triplet Loss (for metric learning/Siamese networks), Focal Loss (for class imbalance), and custom loss functions with regularization penalties (L1/L2).
* **Readings:**
    * Paper: *Focal Loss for Dense Object Detection* (Lin et al., 2017).
* **Videos:**
    * <a href="https://www.youtube.com/watch?v=d2XB5-tuCWU" target="_blank" rel="noopener noreferrer">DeepLearning.AI - Triplet Loss (Andrew Ng)</a>
* **Practical Task:** Implement a custom loss function in PyTorch that combines standard Cross-Entropy with an L1 penalty on the weights. Train a model and evaluate its sparsity and generalization ability.

## Part V: Integration, Scalability, and Critical Evaluation
**Mapping:** CP5, OA5, OA6

### Week 9: Evaluation, Stopping Criteria, and Model Tuning
* **Concepts:** Early stopping, learning rate schedulers (Cosine Annealing, StepLR), comparing methods statistically, analyzing test/validation gaps. Scalability issues with massive models.
* **Readings:**
    * *Deep Learning* (Chapter 7: Regularization, focusing on Early Stopping).
* **Videos:**
    * <a href="https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html" target="_blank" rel="noopener noreferrer">PyTorch Tutorials - Optimizing Model Parameters</a>
* **Practical Task:** Set up an automated hyperparameter tuning pipeline using **Optuna** or **Ray Tune** to find the best optimizer, learning rate, and batch size for a deep neural network, constrained by a 1-hour compute budget.

### Week 10: Capstone / Practical Case Discussion
* **Concepts:** Bringing it all together. Justifying the choice of optimizer, cost function, and hyperparameter tuning strategy based on practical constraints (time, memory/VRAM, dataset size).
* **Practical Task (Final Project):**
    * Select a complex real-world dataset (e.g., from Kaggle).
    * Choose an appropriate deep learning architecture.
    * Perform a rigorous comparison of at least 3 optimization strategies (e.g., SGD+Momentum, Adam, AdamW with Cosine Annealing).
    * Evaluate the impact of two different cost functions or regularization techniques.
    * **Deliverable:** A technical report containing convergence plots, justification of methods chosen considering scalability, and proposals for further improvements (OA5, OA6).

## Core Textbooks & Resources Summary
1. **Nocedal, J., & Wright, S. J.** (2006). *Numerical Optimization*. Springer. (Best for CP1).
2. **Luke, S.** (2013). *Essentials of Metaheuristics*. (Best for CP2).
3. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. (Essential for CP3, CP4).
4. **Frameworks:** PyTorch or TensorFlow, SciPy, DEAP (for genetic algorithms), Optuna.
