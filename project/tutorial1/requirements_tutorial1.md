# **Computational Optimization - Tutorial #1**

## **1. Implementation Suggestions**

The project should address **optimization problems in AI applications**. Below is a comprehensive list of potential areas, categorized by their origin:

### **Applications from Lecture 1 Slides**

- **Supervised Learning Training**
  Minimizing loss functions (e.g., mean squared error, cross-entropy, hinge loss) as the foundation for regression, classification, and neural networks.

- **Linear Regression and Least Squares**
  Solving convex least squares problems to fit model coefficients.

- **Support Vector Machines (SVM)**
  Maximizing class margins via convex quadratic programming with constraints.

- **Computational Advertising and Algorithmic Auctions**
  Optimizing revenue, relevance, or utility under budget, fairness, and ranking constraints.

- **Logistic Regression**
  Minimizing log-loss/cross-entropy using gradient-based methods.

- **Sequential Planning**
  Solving multi-step decision problems to find optimal action sequences.

- **Robot Trajectory Optimization**
  Finding trajectories that minimize energy, time, collision risk, or error under physical and kinematic constraints.

- **Computer Vision**
  Formulating segmentation, reconstruction, pose estimation, calibration, and matching as geometric/photometric error minimization problems.

### **Additional Applications**

- **L1, L2, and Elastic Net Regularization**
  Adding penalties to the objective function to control overfitting; L1 induces sparsity, while L2 penalizes large weights.

- **Feature Selection**
  Selecting relevant subsets of variables; can be framed as combinatorial optimization or sparse penalization.

- **Dimensionality Reduction**
  Methods like PCA and autoencoders minimize reconstruction error or maximize explained variance.

- **Clustering**
  K-means minimizes the sum of squared distances to centroids; other methods optimize separation/coherence objectives.

- **Deep Neural Network Training**
  Non-convex, high-dimensional optimization solved via SGD, Adam, or stochastic gradient variants.

- **Stochastic Gradient Descent (SGD) and Variants (Adam, Adaptive Optimization)**
  Iterative optimization using samples or mini-batches to scale training to large datasets.

- **Hyperparameter Optimization**
  Searching for optimal values (e.g., learning rate, depth, regularization) using grid search, random search, or Bayesian optimization.

- **Bayesian Optimization**
  Global optimization of expensive-to-evaluate functions using surrogate models and acquisition functions. Common in AutoML and model tuning.

- **AutoML and Pipeline Selection**
  Automatically selecting preprocessing, models, architectures, and hyperparameters. Optimization over mixed spaces (continuous, discrete, combinatorial).

- **Neural Architecture Search (NAS)**
  Optimizing network architecture (e.g., layers, connections, operations) using reinforcement learning, evolution, gradients, or Bayesian optimization.

- **Model Compression**
  Reducing computational cost while maintaining performance; a trade-off between accuracy, memory, latency, and energy.

- **Neural Network Pruning**
  Removing irrelevant weights, neurons, or layers to minimize performance loss under size/latency constraints.

- **Neural Network Quantization**
  Approximating weights/activations with lower precision to optimize memory, speed, and energy, especially for edge AI.

- **Knowledge Distillation**
  Training a smaller model to approximate the outputs of a larger model by minimizing a loss between predictive distributions.

- **Reinforcement Learning**
  Maximizing cumulative expected reward; includes Bellman equations, dynamic programming, policy gradients, and optimal control.

- **Policy Optimization**
  Directly adjusting policy parameters to maximize expected return; basis for methods like REINFORCE, actor-critic, and policy gradient.

- **Model Predictive Control (MPC)**
  Repeatedly solving a trajectory optimization problem, executing the first action, and re-optimizing in the next step.

- **SLAM and Robotic Vision**
  Simultaneously estimating camera/robot position and environment structure; typically non-linear least squares.

- **Recommender Systems**
  Matrix factorization minimizes error between observed and predicted ratings; may include regularization and user/item biases.

- **Learning to Rank**
  Optimizing rankings in search, recommendation, or advertising; targets metrics like NDCG, MAP, or differentiable approximations.

- **Word Embeddings**
  Learning vectors that minimize contextual prediction losses, such as negative sampling or noise-contrastive estimation (NCE).

- **Language Models**
  Training via cross-entropy/negative log-likelihood minimization; inference via approximate search for probable sequences.

- **Decoding in NLP (Beam Search)**
  Searching for the most probable token sequence; beam search is a discrete optimization heuristic.

- **Neural Machine Translation**
  Training minimizes sequential loss; generation uses approximate decoding (e.g., greedy search, beam search).

- **Generative Adversarial Networks (GANs)**
  Minimax problem between generator and discriminator; formally a two-player game with a saddle point.

- **Diffusion Models**
  Optimizing objectives related to denoising, score matching, maximum likelihood, and ELBO.

- **Variational Autoencoders (VAEs)**
  Maximizing a variational lower bound (ELBO); involves approximate inference and stochastic optimization.

- **Adversarial Attacks**
  Finding small perturbations that maximize model error; constrained optimization around original data.

- **Adversarial Training and Robustness**
  Min-max problem: minimizing loss in the worst case within an admissible perturbation ball.

- **Counterfactual Explanations**
  Finding the smallest change in input attributes that alters the model’s decision; optimization with plausible constraints.

- **Optimization-Based Explainable AI**
  Finding sparse explanations, local rules, representative examples, or minimal changes to justify predictions.

- **Fairness-Constrained Machine Learning**
  Training models with fairness constraints (e.g., statistical parity, equal opportunity, group-wise bounds).

- **Accuracy–Fairness Trade-off**
  Multi-objective problem: maximizing predictive performance while reducing bias.

- **Differentially Private Optimization**
  Adapting SGD/Adam with noise and clipping to limit individual information leakage.

- **Federated Learning**
  Distributed optimization: multiple clients update models locally and aggregate gradients/parameters without centralizing data.

- **Multi-Objective Optimization in AI**
  Balancing multiple criteria (e.g., accuracy, cost, interpretability, energy, fairness, latency) using Pareto fronts.

- **Edge AI Optimization**
  Optimizing models to run on devices with limited memory, energy, and computational capacity.

- **Inference Optimization**
  Reducing latency and execution cost via batching, quantization, pruning, graph compilation, and hardware selection.

- **Semi/self-Supervised Learning**
  Building auxiliary objectives (e.g., reconstruction, contrast, masked prediction, representation alignment).

- **Contrastive Learning**
  Optimizing a function that pulls similar examples closer and pushes dissimilar examples apart in latent space.

- **Probabilistic Models and Variational Inference**
  Transforming intractable probabilistic inference into an optimization problem over approximate distributions.

- **Differentiable Programming**
  Incorporating optimization, simulation, or decision modules into gradient-trainable networks.

## **2. Project Requirements**

The project must:

- **Focus on Applied Optimization in AI**: Demonstrate the ability to *formulate* optimization problems (not just apply algorithms) in an AI context.
- **Integrate Theory and Practice**: Combine theoretical knowledge with hands-on skills in modeling, implementation, and critical analysis.
- **Collaborative Work**: Be developed in groups of **2 students**, with **equal and active participation** in all phases (including tutorials).
- **Grading**:
  - Minimum passing grade: **9.5/20**.
  - Contributes **80%** to the final course grade.
  - The remaining **20%** comes from **2 individual online quizzes** (10% each).

## **3. Project Objectives**

The project must achieve the following:

1. **Formulate** an optimization problem in an AI context.
2. **Select and implement** appropriate optimization methods.
3. **Analyze** convergence, stability, and performance of the solutions.
4. **Compare** different approaches and **justify** methodological choices.
5. **Propose** well-founded improvements based on results.

## **4. Tutorials and Deadlines**

- **Tutorial #1: Work Plan**
  - Submit a **3–4 page document** outlining:
    - Problem identification and formulation.
    - Optimization methods/techniques to be used.
  - **Deadline**: **May 18, 2026 (23:59)**.

## **5. Submission Formats**

Choose **one** of the following options for project delivery:
- **Option A**: **Jupyter Notebook** containing text, code, and results.
- **Option B**: **Report (15–20 pages)** with code provided as an appendix.
