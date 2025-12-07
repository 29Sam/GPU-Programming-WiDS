### *Accelerating Scientific & AI Workflows with CUDA & Triton*

*A 6-week winter course for researchers and engineers*

---

Most researchers and engineers already have heavy Python code—simulations, numerical solvers, matrix workloads, data transforms, ML ops—that run slowly on CPUs.
This course teaches you to **accelerate your own real project** using **GPU programming** with **CUDA** and **Triton**.

Over 6 weeks, we will go from **GPU intuition → CUDA fundamentals → memory optimization → real kernels → Triton → a final mini-project**, where you implement and benchmark a GPU-accelerated version of a problem you personally care about.

---

## Goals:

By the end of this course, you will be able to:

* Understand *when and why* GPUs outperform CPUs
* Map your compute to the CUDA execution model (threads, blocks, grids)
* Use the GPU memory hierarchy effectively (global, shared, constant memory)
* Write and optimize CUDA kernels for math-heavy workloads
* Use **Triton** to write concise, high-performance GPU kernels in Python
* Profile your compute and quantify speedups vs CPU implementations
* Accelerate a real Python-based research problem—or your own project

---

## Who This Course Is For

* Researchers (CS, physics, engineering, biology, economics, etc.)
* Engineers and developers with compute-heavy workloads
* ML practitioners building custom layers or operations
* Anyone who wants their Python code to run **10×–100× faster**

**Prerequisites:**

* Python and NumPy (required)
* Some C/C++ helpful but **not required**
* Basic linear algebra knowledge

---

## Course Structure (6 Weeks)

Each week contains:

* 📘 **Concepts & readings** from NVIDIA docs and academic lectures
* 💻 **Hands-on coding assignments**
* 📊 **Weekly submission**
* 🧪 **Progress towards the mini-project**

Weekly breakdown:

1. **GPU Intuition & Compute Foundations**
2. **Parallel Thinking & CUDA Basics**
3. **Memory Hierarchy & Performance Optimization**
4. **Real Kernels: GEMM, softmax, compute patterns**
5. **Modern GPU Programming with Triton**
6. **Mini-Project: Accelerate Your Own Research Code**

Each week will have its own markdown file under `weekX.md`.

---

## Final Mini-Project

You will:

1. Choose a real compute-heavy Python task from your research or interests
2. Profile the CPU implementation
3. Rewrite the bottleneck(s) using **CUDA and/or Triton**
4. Benchmark and visualize the speedup
5. Submit a short write-up or slide deck summarizing the problem, GPU approach, and results

Examples of suitable problems:

* PDE or simulation step (finite-difference, Monte-Carlo, particle update)
* Numerical algorithms (scan, reduction, iterative solvers)
* Custom ML ops (attention block, loss function, activation)
* Data transformations (pairwise distances, feature extraction, filtering)

You finish the course with:

* A **GPU-accelerated version** of something meaningful
* A clean codebase
* Techniquees you can use in research papers or engineering work

---

## Tools You Will Use

* **CUDA Toolkit** and the official **CUDA Programming Guide**
* **Python + PyTorch/NumPy**
* **Triton** for high-level GPU programming
* **NVCC**, **Nsight Systems**, **Nsight Compute**, and timing via `torch.cuda.Event`
* Git & GitHub for submissions

---

## 📂 Repository Structure

Gpu-Programming-WiDS/
│
├── README.md                     # Main course description
│
├── week1/
│   ├── README.md                 # Instructions, learning goals, resources
│   ├── assignment.md             # Weekly assignment
│   └── resources/                # Optional PDFs, notes, papers for this week
│       └── ...
│
├── week2/
│   ├── README.md
│   ├── assignment.md
│   └── resources/
│       └── ...
│
├── week3/
│   ├── README.md
│   ├── assignment.md
│   └── resources/
│       └── ...
│
├── week4/
│   ├── README.md
│   ├── assignment.md
│   └── resources/
│       └── ...
│
├── week5/
│   ├── README.md
│   ├── assignment.md
│   └── resources/
│       └── ...
│
├── week6/
│   ├── README.md                 # Mini-project instructions
│   ├── assignment.md             # Final project specification
│   └── resources/
│       └── ...
│
└── shared-resources/             # CUDA docs, lecture links, cheat sheets, etc.
    ├── links.md
    └── papers.md


---

## Key References

* **NVIDIA CUDA Programming Guide:**
  [https://docs.nvidia.com/cuda/cuda-programming-guide/](https://docs.nvidia.com/cuda/cuda-programming-guide/)
* **CUDA Samples:**
  [https://github.com/NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)
* **GPU Gems 3 (Scan chapter):**
  [https://developer.nvidia.com/gpugems/gpugems3](https://developer.nvidia.com/gpugems/gpugems3)
* **Triton (OpenAI):**
  [https://github.com/openai/triton](https://github.com/openai/triton)

---

## 🚀 Getting Started

1. Fork this repository
2. Ensure access to an NVIDIA GPU (local or cloud)
3. Install the CUDA Toolkit + Python environment
4. Start with **Week 1 → `weeks/week1.md`**

---
