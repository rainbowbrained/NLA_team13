# NLA_team13: Analyzing the Evolution of Weight Matrices in the OPT-13B LLM During Training

This project focuses on the analysis of weight matrices in the pythia-410M Large Language Model (LLM) and its training checkpoints.

We investigate the matrices associated with the attention mechanism, including Key, Query, Value, and Output Projection matrices, as well as the token embedding matrix and the weight matrices of the MLP for each layer and for every training checkpoint.

Our analysis will explore how the characteristics of these matrices evolve during the training process through key metrics:
examination of their spectral properties
1. assessment of matrices orthogonality
2. evaluation of sparsity across all matrices
3. analysis matrices weights evolution in different matrix norms
By employing a comprehensive approach to these characteristics, we aim to enhance our understanding of LLM training dynamics and contribute to discussions on model interpretability and optimization. This work builds on existing literature that emphasizes the importance of understanding weight distribution properties during training, providing empirical insights that can inform future research in model architecture and training methodologies.
