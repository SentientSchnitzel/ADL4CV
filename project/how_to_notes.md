## **Task 1: Implement and Train a Diffusion Model on CheXpert**
### **Goal:**  
- Train a **diffusion model** on the **CheXpert dataset** (a large chest X-ray dataset) to generate medical images.
- Train it **with and without classifier-free guidance** to compare how conditioning affects image generation.

### **Steps:**
1. **Choose a Diffusion Model Architecture:**  
   - You might use **DDPM** (Denoising Diffusion Probabilistic Models) or a variant like **Stable Diffusion**.
   - Consider using **U-Net** as the backbone for noise prediction.

2. **Prepare the CheXpert Dataset:**  
   - Load and preprocess the dataset (normalize images, resize if necessary).
   - Create labeled conditioning based on **disease labels** (e.g., "Pneumonia", "No Finding", etc.).

3. **Implement Classifier-Free Guidance (CFG):**  
   - CFG allows the model to generate images based on disease labels.
   - During training, randomly drop labels **50% of the time** and use both **conditioned** and **unconditioned** noise predictions:
     $$
     \epsilon_{\theta}(x_t, y) = \epsilon_{\text{uncond}}(x_t, t) + w \cdot (\epsilon_{\text{cond}}(x_t, y) - \epsilon_{\text{uncond}}(x_t, t))
     $$
   - Compare training with **CFG enabled vs. disabled**.

4. **Train the Model:**  
   - Use a dataset split (train/validation/test).
   - Optimize with **AdamW** and a loss function like **MSE** on the noise prediction.
   - Log loss values and visualize generated images.

---

## **Task 2: Implement and Validate Testing for Memorization**
### **Goal:**  
- Implement the **memorization detection method** from the paper.
- Validate whether high memorization scores correspond to images that look **identical** to training data.

### **Steps:**
1. **Extract Text-Conditional Noise Predictions:**  
   - Modify your trained diffusion model to **record noise predictions** at each time step.
   - Compute the **magnitude of text-conditional noise prediction**:
     $$
     M_t = \| \epsilon_{\text{text}}(x_t, t, y) \|
     $$
   - Measure this across all time steps.
   - Note: you have a balance between noise removal with and without classifier-free guidance (text)
     -  $\epsilon_{\theta}(x_t, t, y) = \epsilon_{\text{uncond}}(x_t, t) + w \cdot \epsilon_{\text{text}}(x_t, t, y)$
     -  Where high magnitude of $\epsilon_{\text{text}}$ indicates memorization

2. **Detect Memorized Prompts:**  
   - If **$M_t$** is consistently high across time steps, mark the prompt as a potential memorization case.
   - Compare with known **training images** using a similarity metric (e.g., SSIM, LPIPS, or CLIP similarity).
     - I.e., show that high-memorization cases produce images **identical** or highly similar to training data - found using **CLIP similarity**.

3. **Validate Results:**
   - **Visual Inspection:** Check if high-memorization cases produce **identical** or highly similar images to training data.
   - **Semantic Similarity Metrics:** Use tools like **CLIP** to compare generated vs. training images.
     - I.e. show this by plotting the **CLIP similarity** between generated images and training images using e.g. **t-SNE**.

---

## **Task 3: Mitigate Memorization**
### **Goal:**  
- Implement **techniques to reduce memorization** and **retest** the model.

### **Steps:**
1. **Modify Training to Reduce Memorization:**
   - **Data Augmentation:** Increase image diversity by adding slight **rotations, noise, contrast changes**, etc.
   - **Differential Privacy (DP):** Introduce **Gaussian noise** into gradients during training.
   - **Reduced Guidance Strength:** Lower the classifier-free guidance weight **$w$** to make the model rely less on explicit conditioning.

2. **Modify Prompts to Avoid Trigger Tokens:**  
   - Implement **gradient-based token significance detection**:
     - Compute token **Significance Score (SS)** as:
       $$
       SSe_i = \frac{1}{T} \sum_{t=1}^{T} \| \nabla_{e_i} L(x_t, e) \|^2
       $$
     - Identify **high-importance tokens** and experiment with **removing or replacing them**.

3. **Validate Again:**  
   - Repeat the memorization test after mitigation.
   - Compare metrics before and after mitigation to **quantify improvements**.




