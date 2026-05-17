# Project — Computational Optimization in AI

## Project Framework

Following a Problem-Based Learning approach, the assessment of the Computational Optimization course unit (UC) has as its main element the development of an applied project focused on the formulation and resolution of an **optimization problem in the AI applied context**. The project:

- **Assesses the capacity to think in terms of optimization applied to AI and not merely to apply algorithms**, based on the objectives listed below;
- Must demonstrate the integration of theoretical knowledge with practical competences in modelling, implementation, and critical analysis;
- Is developed as group work, with **2 students per group**, in which both students must have **active participation in all phases/tutorials**;
- Has a **minimum grade of 9.5** (scale 0-20) and contributes to **80% of the final grade** in the course unit (note: the remaining 20% (= 2×10%) corresponds to the 2 individual online quizzes).

## Project Objectives

- Formulate an optimization problem in the AI context
- Select and implement optimization methods
- Analyse convergence, stability, and performance
- Compare approaches and justify choices
- Propose duly substantiated improvements

## Project Theme

Each group of 2 students must select an application from the list below or, alternatively, propose an application subject to approval.

## Project Phases and Key Dates

- **Phase 1 (1st tutorial)** (20% of the course unit grade): work plan (3-4 pages) with identification and formulation of the optimization problem and the optimization methods/techniques to be used.
  **Submission deadline: 18 May (by 23:59)**

- **Phase 2 (2nd tutorial)** (60% of the course unit grade): final project with implementation of the methods, experimental results, critical performance analysis, comparison of the applied methods, and improvement proposals. Phase 2 concludes with a defence of the work developed.
  **Submission deadline: 11 June (by 23:59)**
  **Online defence: date to be defined within the period 16-18 June**

## Project Delivery Formats

- **Option A:** Jupyter Notebook with text, code, and results.
- **Option B:** Report (15-20 pages) with code attached as an annex.

---

## List of AI Application Proposals

### Applications already included in Class 1 slides

- **Supervised learning training**
  Minimize a loss function: squared error, cross-entropy, hinge loss. It is the base formulation of regression, classification, and neural networks.

- **Linear regression and least squares**
  Solve a least-squares problem, often convex, to fit coefficients.

- **Support vector machines (SVM)**
  Maximize the margin between classes, generally through convex quadratic programming with constraints.

- **Computational advertising and ad auctions**
  Maximize revenue, relevance, or utility under budget, fairness, and ranking constraints.

- **Logistic regression**
  Minimize log-loss / cross-entropy, generally with gradient-based methods.

- **Sequential decision-making / planning**
  Solve a multi-step decision problem where each action changes the state, and the optimization process seeks the optimal sequence.

- **Robot trajectory optimisation**
  Find trajectories that minimize energy, time, collision risk, or error, under physical and kinematic constraints.

- **Computer vision**
  Segmentation, reconstruction, pose estimation, calibration, and matching can be formulated as minimization of geometric/photometric error.

### Applications not included in Class 1 slides

- **L1, L2 and Elastic Net regularisation**
  Add penalty terms to the objective function to control overfitting; L1 induces sparsity and L2 penalizes large weights.

- **Feature selection**
  Choose subsets of relevant variables; can be viewed as combinatorial optimization or as sparse penalization.

- **Dimensionality reduction**
  PCA, autoencoders, and related methods minimize reconstruction error or maximize explained variance.

- **Clustering**
  K-means minimizes the sum of squared distances to centroids; other methods use separation/coherence objectives.

- **Deep neural network training**
  Non-convex, high-dimensional problem solved by SGD, Adam, and other stochastic gradient variants.

- **Stochastic gradient descent and variants (SGD, Adam, adaptive optimisation)**
  Iterative optimization using samples or mini-batches to scale training to large datasets.

- **Hyperparameter optimisation**
  Search for optimal values for learning rate, depth, regularization, number of trees, etc. Uses grid search, random search, Bayesian optimization.

- **Bayesian optimisation**
  Global optimization of expensive-to-evaluate functions, using surrogate models and acquisition functions. Widely used in AutoML and model tuning.

- **AutoML and automated pipeline selection (automated machine learning)**
  Automatically choose pre-processing, model, architecture, and hyperparameters. It is optimization over mixed spaces: continuous, discrete, and combinatorial.

- **Neural architecture search (NAS)**
  Optimize the architecture of a network: number of layers, connections, blocks, operations. Uses RL, evolution, gradient methods, and Bayesian optimization.

- **Model compression**
  Reduce computational cost while maintaining performance: a trade-off problem between accuracy, memory, latency, and energy.

- **Neural network pruning**
  Remove weights, neurons, or layers of low relevance, minimizing performance loss under size or latency constraints.

- **Neural network quantisation**
  Approximate weights and activations by lower-precision values; optimizes memory, speed, and energy, especially in embedded AI / edge AI.

- **Knowledge distillation**
  Train a small model to approximate the outputs of a large model, minimizing a loss between predictive distributions.

- **Reinforcement learning**
  Maximize expected cumulative reward; includes Bellman equations, dynamic programming, policy gradient, and optimal control.

- **Policy optimisation**
  Directly adjust the parameters of a policy to maximize expected return; the basis of methods such as REINFORCE, actor-critic, and policy gradient.

- **Model predictive control (MPC)**
  Repeatedly solve a trajectory optimization problem, execute the first action, and re-optimize at the next step.

- **SLAM and bundle adjustment**
  Simultaneously estimate camera/robot position and the structure of the environment; typically nonlinear least squares.

- **Recommender systems**
  Matrix factorization minimizes the error between observed and predicted ratings; may include regularization and user/item biases.

- **Learning to rank**
  Optimize rankings in search, recommendation, or advertising; ideally optimize metrics such as NDCG, MAP, or differentiable approximations.

- **Word embeddings**
  Learn vectors that minimize contextual prediction losses, such as negative sampling or NCE.

- **Language models**
  Training by minimization of cross-entropy / negative log-likelihood; inference by approximate search of likely sequences.

- **Decoding in NLP (beam search)**
  Search for the token sequence with the highest approximate probability; beam search is a heuristic for discrete optimization.

- **Neural machine translation**
  Training minimizes a sequential loss; generation uses approximate decoding such as greedy search or beam search.

- **Generative adversarial networks (GANs)**
  Minimax problem between generator and discriminator; formally a two-player game with a saddle point.

- **Diffusion models**
  Optimize objectives related to denoising, score matching, maximum likelihood, and ELBO.

- **Variational autoencoders (VAEs)**
  Maximize a variational lower bound (ELBO); involves approximate inference and stochastic optimization.

- **Adversarial attacks**
  Find small perturbations that maximize the model's error; constrained optimization around the original data.

- **Adversarial training and robustness**
  Min-max problem: minimize the worst-case loss within a ball of admissible perturbations.

- **Counterfactual explanations**
  Search for the smallest change in input attributes that changes the model's decision. A constrained optimization problem with plausibility constraints.

- **Optimisation-based explainable AI**
  Find sparse explanations, local rules, representative examples, or minimal changes that justify a prediction.

- **Fairness-constrained machine learning**
  Train models with equity constraints: statistical parity, equality of opportunity, group-level bounds.

- **Accuracy–fairness trade-off**
  Multi-objective problem: maximize predictive performance while simultaneously reducing bias.

- **Differentially private optimisation**
  Adapt SGD/SGDA with noise and clipping to limit leakage of individual information.

- **Federated learning**
  Distributed optimization: multiple clients update models locally and aggregate gradients/parameters without centralizing data.

- **Multi-objective optimisation in AI**
  Balance several criteria: accuracy, cost, interpretability, energy, fairness, latency. Uses Pareto fronts.

- **Edge AI optimisation**
  Optimize models to run on devices with limited memory, energy, and computational capacity.

- **Inference optimisation**
  Reduce latency and execution cost: batching, quantization, pruning, graph compilation, and hardware choice.

- **Semi-supervised and self-supervised learning**
  Construct auxiliary objectives: reconstruction, contrast, prediction of hidden parts, alignment of representations.

- **Contrastive learning**
  Optimize a function that draws similar examples together and pushes dissimilar examples apart in the latent space.

- **Probabilistic models and variational inference**
  Transform difficult probabilistic inference into an optimization problem over approximate distributions.

- **Differentiable programming**
  Incorporate optimization, simulation, or decision modules within networks trainable by gradient.
