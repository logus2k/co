### **1. Formulate an Optimization Problem**

* **Current Status:** Partial. The code sets up the environment, data, and model perfectly.
* **What’s Needed:** To fulfill the "integrate theory and practice" requirement, you need to add Markdown cells before the code blocks. You must explicitly write out the mathematical formulation of the non-convex optimization problem, detailing the loss function (MSE) and the parameters being optimized (the MLP weights).

### **2. Select and Implement Methods**

* **Current Status:** Strong. Implementing `MyAdam` from scratch is exactly what the professors are looking for. It proves you aren't just treating PyTorch as a black box.
* **What’s Needed:** To fulfill the comparative aspect of the prompt, you need to implement at least one or two other custom optimizers (like `MySGD` with momentum) alongside `MyAdam` to show how different strategies navigate the loss landscape.

### **3. Analyze Convergence, Stability, and Performance**

* **Current Status:** Good start. You are capturing the loss history and plotting a basic convergence curve (Cell 11), as well as calculating the final PSNR (Cell 14).
* **What’s Needed:** The requirements demand an analysis of *stability*. With the local compute power you have available, expanding this training loop to run across multiple random seeds or executing learning rate sweeps will be fast. You need to plot these multiple runs to show variance and prove the optimizer's stability.

### **4. Compare Approaches and Justify Choices**

* **Current Status:** Missing. The PoC currently only demonstrates one approach (NeRF + Adam + MSE).
* **What’s Needed:** This is the core of the 15-20 page equivalent deliverable. You need to add the 3D Gaussian Splatting baseline to compare implicit vs. explicit representations. You also need to compare the convergence plots of `MyAdam` vs. `MySGD`, and justify *why* one performed better than the other on this specific 3D reconstruction task.

### **5. Propose Improvements**

* **Current Status:** Missing.
* **What’s Needed:** A final "Discussion and Future Work" Markdown section at the end of the notebook. You must critically evaluate your results and propose well-founded algorithmic or architectural improvements based on what you observed (e.g., suggesting learning rate restarts or adaptive view sampling if your SGD model gets stuck in local minima).

### **6. Delivery Format**

**The Verdict:** The remaining work is primarily structural and experimental: wrapping your existing code in loops to test different conditions, adding the 3DGS comparison, and writing the analytical text.

