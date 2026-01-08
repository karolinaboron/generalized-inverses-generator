
# Generalized Matrix Inverse Generator 
### Engineering Thesis Project 
**Topic:**  Pseudoinverses of a matrix and selected applications
**Author:** Karolina Boroń
**University:** Cracow University of Technology

## Live Demo
The application is deployed and available online:  
**[Click here to open the App](https://generalized-inverses-generator.streamlit.app/)**

##  About the Project
This repository contains the source code for a web application developed as part of an Engineering Thesis. The main goal of the project was to implement algorithms for calculating various classes of **generalized matrix inverses**.

Unlike standard calculators that only provide the Moore-Penrose pseudoinverse ($A^+$), this tool allows users to generate matrices satisfying specific subsets of **Penrose conditions**:

1. $AXA = A$
2. $XAX = X$
3. $(AX)^* = AX$
4. $(XA)^* = XA$

### Key Features
* **Symbolic & Numerical Computation:**
    * **Symbolic Mode (General Solution):** Derives the **general parametric formula** for the entire set of solutions. Users can analyze the structure of the solution space and substitute specific values for the free parameters.
    * **Numerical Mode (Random Instance):** Generates a random, specific matrix example that satisfies the selected axioms.
* **Step-by-step solutions:** Visualization of intermediate steps.
* **Verification:** Automatic check if the result satisfies the required Moore-Penrose equations.

## 🛠️ Tech Stack
* **Language:** Python 3.11.9+
* **Framework:** Streamlit
* **Libraries:** NumPy, SymPy, Pandas

