# The Path to a World-Class ML Engineer

This document outlines the strategic roadmap to transition from a graduating Master's student and ML Intern at Bloomreach to a highly respected, "Europe-class" or "World-class" AI/ML Systems Engineer.

The overarching philosophy is to move up the abstraction stack: from merely training models in isolation to architecting, deploying, optimizing, and maintaining large-scale, resilient AI systems. 

## The Baseline (Year 0 - Graduation & Full-Time Entry)

*   **Current Status:** Graduating from Masaryk University (Brno), moving to full-time at Bloomreach (Slovakia/Remote).
*   **Current Salary (Part-Time):** ~1,200€ Brutto (approx. 1,000€ Net for 50% capacity).
*   **Expected Salary (Full-Time Junior ML Engineer - SK/CZ Market):** ~2,400€ - 3,000€ Brutto. (If you get ~2.5k€, that's a solid standard entry for a smart grad in the region. Top-tier companies might push 3.5k€).
*   **Immediate Goal:** Survive the transition, learn production engineering, and establish a reputation as a reliable, delivery-focused engineer.

---

## Phase 1: The Production Engineer (Months 1-18)
*The goal here is to become undeniably useful. Stop acting like a student and start acting like an engineer who ships.*

### Core Focus: MLOps, Data Engineering, and Systems
You need to master how ML models survive in the real world. 

**Buzzwords & Technologies to Master:**
*   **Containerization & Orchestration:** Docker, Kubernetes (K8s), Helm charts, Containerd. (Understand how a model is actually served in a pod, resource limits, and auto-scaling).
*   **CI/CD for ML & Infrastructure as Code (IaC):** GitHub Actions, GitLab CI, Terraform, Ansible.
*   **Model Serving (Basic & Advanced):** FastAPI, TorchServe, Triton Inference Server, Seldon Core, KServe.
*   **Workflow Orchestration:** Airflow, Prefect, Dagster, Kubeflow Pipelines.
*   **Data Processing & Feature Stores:** Spark, Kafka, Flink, dbt (data build tool), Feast or Hopsworks (for feature stores).
*   **Tracking & Registry:** MLflow, Weights & Biases (W&B), DVC (Data Version Control).
*   **Monitoring (Drift/Performance) & Observability:** Prometheus, Grafana, Datadog, Evidently AI, Arize.
*   **Cloud Platforms:** AWS (SageMaker, EC2, EKS, S3), GCP (Vertex AI, GKE), or Azure ML. Pick one and know it deeply.

### Milestones
*   **Month 3:** Independently deploy a model from training to a staging environment using the company's CI/CD pipeline.
*   **Month 6:** Set up comprehensive monitoring for a production model (latency, throughput, and data drift).
*   **Month 12:** Lead a project to optimize an existing data pipeline or model serving endpoint, demonstrating a measurable reduction in latency or cost.
*   **Month 18:** Mentoring the next batch of interns. You are now a solid "Mid-Level" ML Engineer.

**Expected Salary Progression (Mid-Level in SK/CZ/Remote Europe):** ~3,500€ - 5,000€ Brutto.

---

## Phase 2: The Specialization & Depth (Years 1.5 - 3)
*You are now a solid engineer. To become "world-class," you must build extreme depth in a high-value niche.*

### Option A: The Inference & Performance Guru (High Demand, High Complexity)
Making large models run fast and cheap.
*   **Buzzwords:** CUDA C++, Triton (OpenAI's kernel language), vLLM, TensorRT, TensorRT-LLM, ONNX Runtime, OpenVINO, Quantization (AWQ, GPTQ, GGUF, FP8, INT4), KV-Cache optimization (PagedAttention, FlashAttention), Speculative Decoding.
*   **Goal:** Be the person who can take a 70B parameter model and cut its serving cost by 40% without losing quality.

### Option B: The Agentic & RAG Architect (High Visibility, Product Focus)
Building robust reasoning systems around LLMs.
*   **Buzzwords:** Agentic Workflows, LangGraph, AutoGen, CrewAI, Advanced RAG (HyDE, Parent Document Retriever, Semantic Routing, GraphRAG), Vector Databases (Pinecone, Qdrant, Milvus, Weaviate), Evaluation Frameworks (Ragas, TruLens, LangSmith), Prompt Engineering (few-shot, Chain-of-Thought, ReAct).
*   **Goal:** Build systems that don't just "talk," but reliably execute complex, multi-step business logic with low hallucination rates.

### Option C: The Distributed Training Expert (Niche, Foundation Model Focus)
Training massive models across GPU clusters.
*   **Buzzwords:** DeepSpeed, Megatron-LM, FSDP (Fully Sharded Data Parallel), Ray, NCCL, MPI, Slurm, JAX, TPU programming, Ring-AllReduce.
*   **Goal:** Understand how to orchestrate a training run across 512 GPUs without the cluster crashing every 3 hours.

### Option D: Edge AI & Embedded Systems (Robotics, IoT, Self-Driving)
*Does it make sense to learn electronics? YES, if you want to enter the physical world.*
Bringing AI to hardware with strict power and memory constraints. If you have an interest in robotics (like Pepper robots) or autonomous vehicles, this is an incredibly lucrative and highly-defensible moat. Pure software engineers struggle here.
*   **Buzzwords:** C/C++, Rust, ROS 1/2 (Robot Operating System), Edge TPU, NVIDIA Jetson (JetPack), CoreML, TensorFlow Lite, Microcontrollers (ESP32, STM32), RTOS (Real-Time Operating Systems), Sensor Fusion (LiDAR, Camera, IMU), SLAM (Simultaneous Localization and Mapping).
*   **Goal:** Successfully deploy a real-time object detection or tracking model on an edge device (like a Jetson Nano or a robot) running at 30+ FPS on a battery.

### Milestones
*   **Year 2:** Become the recognized "go-to" person in your department for your chosen niche. 
*   **Year 2.5:** Publish a deep-dive technical blog post (either on the company blog or personal) detailing a complex problem you solved.
*   **Year 3:** Make your first meaningful contributions to a major open-source project (e.g., a PR to vLLM, LangChain, or Hugging Face Transformers).

**Expected Salary Progression (Senior/Specialist - Remote Europe/Top Tier Local):** 5,000€ - 8,000€+ Brutto. (If working remote for US/UK/Swiss companies, this can scale significantly higher: €80k - €150k+/year).

---

## Phase 3: The "World-Class" Recognition (Years 3 - 5)
*Being world-class means your impact extends beyond your employer. You are recognized by your peers globally.*

### Core Focus: Open Source, Architecture, and Thought Leadership

*   **Read Code, Not Just Papers:** Start reading the source code of the tools you use. Understand how `vLLM` manages memory or how `PyTorch` implements autograd.
*   **Build in Public:** Maintain an active GitHub profile. Open-source your utility scripts, create minimal reproducible examples for complex architectures, or maintain a popular library.
*   **Conferences & Networking:** Speak at major European AI conferences (e.g., PyData, KDD, local AI summits). Don't just attend; present your work on scaling or optimizing ML systems.
*   **Systems Architecture:** Move from optimizing single components to designing the entire AI infrastructure for a product line.

### Milestones
*   **Year 4:** Become a core contributor or highly active maintainer for a recognized open-source AI project.
*   **Year 4.5:** Speak at a major international conference.
*   **Year 5:** Attain a "Staff ML Engineer" or "Principal ML Engineer" level. You are now architecting solutions that define the technical direction of the company.

**Expected Salary Progression (Staff/Principal - Global Remote):** €120k - €250k+ per year (often heavily weighted with equity/RSUs).

---

## Phase 4: The Mathematical Moat (PhD-Level Depth)
*To build the state-of-the-art rather than just consume it, you must fluently speak the language of the algorithms.*

While MLOps pays the bills today, profound mathematical intuition is what allows you to debug failing architectures, invent new custom loss functions, and optimize kernels at the hardware level.

### 1. Linear Algebra & Tensor Calculus (The Engine)
You need to move beyond matrix multiplication to understanding vector spaces, manifolds, and how operations map to hardware memory.
*   **Concepts:**
    *   **Eigendecomposition & SVD:** Beyond the formulas—understand them geometrically as transformations and dimensionality reduction (PCA, low-rank adaptations like LoRA).
    *   **Tensor Decompositions:** CP decomposition, Tucker decomposition (crucial for model compression).
    *   **Differential Geometry & Manifolds:** Understanding that deep learning often involves finding lower-dimensional manifolds in high-dimensional spaces (crucial for representation learning and generative models like Normalizing Flows).
    *   **Jacobians & Hessians:** For understanding second-order optimization and stability of training dynamics.
*   **Sources:**
    *   *Linear Algebra and Learning from Data* by Gilbert Strang (accessible but deep).
    *   *Introduction to Smooth Manifolds* by John M. Lee (if you want deep geometric theory).

### 2. Probability Theory & Statistical Mechanics (The Uncertainty)
Machine learning is fundamentally applied statistics. You must understand distributions, not just as formulas, but as representations of uncertainty.
*   **Concepts:**
    *   **Information Theory:** Entropy, Cross-Entropy, Kullback-Leibler (KL) Divergence, Jensen-Shannon Divergence, Mutual Information. (Understand *why* cross-entropy is the right loss for classification).
    *   **Bayesian Inference & Graphical Models:** Markov Chains, Hidden Markov Models (HMMs), Conditional Random Fields (CRFs), Variational Inference (ELBO - Evidence Lower Bound, crucial for VAEs and Diffusion models).
    *   **Stochastic Processes:** Brownian motion, Langevin dynamics (the mathematical foundation of Diffusion models).
    *   **Energy-Based Models:** Understanding the connection between statistical physics (Boltzmann distributions, partition functions) and deep learning.
*   **Sources:**
    *   *Pattern Recognition and Machine Learning* (PRML) by Christopher Bishop (The absolute Bible for Bayesian ML).
    *   *Information Theory, Inference, and Learning Algorithms* by David MacKay.

### 3. Optimization Theory (The Convergence)
Knowing why a model doesn't converge is often more valuable than knowing how to start training.
*   **Concepts:**
    *   **Convex vs. Non-Convex Optimization:** Saddle points, local minima, Lipschitz continuity, and smoothness.
    *   **Stochastic Optimization:** Why SGD works, Adam, RMSprop, AdaFactor, and the theoretical guarantees (or lack thereof) for convergence in over-parameterized regimes.
    *   **Constrained Optimization:** Lagrange multipliers, KKT conditions (essential for SVMs and robust optimization).
    *   **Optimal Transport:** Wasserstein distances (Earth Mover's Distance) – critical for GANs and modern generative models.
*   **Sources:**
    *   *Convex Optimization* by Stephen Boyd and Lieven Vandenberghe.
    *   *Deep Learning* (Goodfellow, Bengio, Courville) - Chapter 8 on Optimization.

### How to Approach Learning This Depth

1.  **Just-in-Time Learning (The "Spiderweb" Method):** Do not sit down and try to read a math textbook cover-to-cover. You will burn out. Instead, read a seminal paper (e.g., the original DDPM paper for diffusion, or the LoRA paper). When you hit an equation you don't understand, pause. Go down the rabbit hole for *that specific equation*. Build your mathematical knowledge outward from practical problems.
2.  **Implement from Scratch:** The ultimate test of mathematical understanding is translating it into code without a library. Implement a basic Autograd engine (like Andrej Karpathy's `micrograd`). Implement backpropagation for a CNN layer using pure NumPy. Implement K-Means or PCA from scratch.
3.  **The "Feynman Technique" for Math:** Write down the concept (e.g., "KL Divergence"). Try to explain it in plain text, then geometrically, then algebraically. If your explanation breaks down, you've found a gap in your knowledge.
4.  **Follow the Giants:** Watch lectures from foundational thinkers. 
    *   *Andrej Karpathy's "Zero to Hero" neural network series.*
    *   *Stanford CS224N (NLP) and CS231N (Vision) lectures.*
    *   *Yannic Kilcher's paper breakdowns on YouTube (excellent for understanding the math behind new architectures).*

---

## The Ultimate Advice for the Journey

1.  **Don't Ignore Software Engineering:** Bad code with good math is still bad code. Learn design patterns, write tests, and understand system architecture. The best ML engineers are simply excellent software engineers who happen to know linear algebra and neural networks.
2.  **Follow the Compute:** The bottleneck in AI right now is compute. Skills that optimize compute (CUDA, distributed systems, efficient inference) command the highest premium.
3.  **Stay Grounded in Business Value:** A 1% increase in an F1 score is useless if it costs $1M to serve. Always align your technical work with what actually makes or saves money for the business.
4.  **Embrace the Churn:** The AI ecosystem reinvents itself every 6 months. Your value isn't in knowing *today's* hottest framework; it's in your ability to learn *tomorrow's* framework in a weekend because your fundamentals are rock solid.
