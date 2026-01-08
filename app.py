import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import string
import re

st.set_page_config(page_title="Generator Odwrotności", layout="wide")

I = sp.I 


def smart_number_format(val, tol=1e-5, max_denom=1000):
    try:
        val_rounded = round(val)
        if abs(val - val_rounded) < 1e-9:
            return sp.Integer(val_rounded)

        candidate = sp.nsimplify(val, tolerance=tol, rational=True)

        if sp.denom(candidate) > max_denom:
            return sp.Float(round(val, 4), 4)

        return candidate

    except Exception:
        return sp.Float(round(val, 4), 4)

def numpy_to_smart_sympy_matrix(arr):
    matrix_data = []
    
    for row in arr:
        row_data = []
        for val in row:
            
            real_part = smart_number_format(np.real(val))
            imag_part = smart_number_format(np.imag(val))
            
            sympy_val = real_part + imag_part * sp.I
            row_data.append(sympy_val)
            
        matrix_data.append(row_data)
        
    return sp.Matrix(matrix_data)

def show_matrix(matrix_obj):
    if hasattr(matrix_obj, 'shape'): 
        latex_code = sp.latex(matrix_obj)
    else:
        st.latex(sp.latex(matrix_obj))
        return

    if "frac" in latex_code:
        latex_code = latex_code.replace(r"\begin{matrix}", r"\begin{matrix} \rule{0pt}{3ex}")
        latex_code = latex_code.replace(r"\end{matrix}", r"\rule[-1.5ex]{0pt}{0pt} \end{matrix}")
        
        rows_count = latex_code.count(r"\\") + 1
        if rows_count <= 2:
            latex_code = latex_code.replace(r"\\", r"\\[0.6em]")
        else:
            latex_code = latex_code.replace(r"\\", r"\\[1.5em]")
            
    st.latex(latex_code)


def clean_input_string(raw_val):
    
    if not raw_val or raw_val.strip() == "":
        return "0"
    
    val = raw_val.replace(",", ".").replace(" ", "").replace("j", "i")
    val = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', val)
    val = val.replace("i", "I")
    
    return val

def convert_cell(val):

    clean_val = clean_input_string(str(val))
    math_obj = sp.sympify(clean_val, locals={'I': sp.I}) 
    return sp.nsimplify(math_obj, tolerance=1e-10, rational=True)

def parse_dataframe_to_matrix(df):
    try:
        matrix_data = []
        
        for m, row in enumerate(df.values):
            parsed_row = []
            for n, val in enumerate(row):
                try:
                    parsed_row.append(convert_cell(val))
                except Exception:
                    return None, f"Nieprawidłowa wartość w wierszu {m+1}, kolumnie {n+1}. Wartość: '{val}'"
            
            matrix_data.append(parsed_row)
            
        return sp.Matrix(matrix_data), None

    except Exception as e:
        return None, f"Błąd krytyczny: {e}"

def prettify_result(X_matrix):

    if not isinstance(X_matrix, sp.Matrix): 
        return X_matrix
    
    X_copy = sp.Matrix(X_matrix)
    surviving_params = sorted(list(X_copy.free_symbols), key=lambda s: s.name)
    
    alphabet = list(string.ascii_lowercase)
    for char in ['i', 'j', 'e', 'f', 'k']: 
        if char in alphabet: 
            alphabet.remove(char)

    final_subs = {}

    for idx, param in enumerate(surviving_params):
        cycle_number = (idx // len(alphabet)) + 1
        char_idx = idx % len(alphabet)
        name = f"{alphabet[char_idx]}_{{{cycle_number}}}"
        final_subs[param] = sp.Symbol(name, complex=True)

    return X_copy.subs(final_subs)

def verify_axioms(A, X):
    results = {}

    if isinstance(X, sp.Matrix):
        
        def check_zero(expr):
            clean_matrix = sp.simplify(expr)
            rows = clean_matrix.shape[0]
            cols = clean_matrix.shape[1]
            return clean_matrix == sp.zeros(rows, cols)

        results[1] = check_zero(A * X * A - A)
        results[2] = check_zero(X * A * X - X)
        results[3] = check_zero((A * X).H - (A * X))
        results[4] = check_zero((X * A).H - (X * A))
        
        return results

    elif isinstance(X, np.ndarray):
        
        if isinstance(A, sp.Matrix): 
             A_np = np.array(A.tolist(), dtype=complex)
        else: 
             A_np = A
        
        def check_zero_np(arr):
            return np.allclose(arr, 0, atol=1e-7)

        AX = A_np @ X
        XA = X @ A_np
        
        results[1] = check_zero_np(A_np @ X @ A_np - A_np)
        results[2] = check_zero_np(X @ A_np @ X - X)
        results[3] = check_zero_np(AX.conj().T - AX)
        results[4] = check_zero_np(XA.conj().T - XA)
        
        return results
    
    else:
        return {k: False for k in range(1, 5)}

def solve_equations(A, selected_axioms):
    m, n = A.shape

    vars_list = []
    G_cells = []
    for i in range(n):
        
        row = []
        for j in range(m):
            r_var = sp.Symbol(f'r_{i+1}_{j+1}', real=True)
            c_var = sp.Symbol(f'c_{i+1}_{j+1}', real=True)
            vars_list.extend([r_var, c_var])
            row.append(r_var + sp.I * c_var)
        G_cells.append(row)

    G = sp.Matrix(G_cells)

    equations = []
    latex_parts = []

    if 3 in selected_axioms:
        AG = A @ G
        diff3 = AG.H - AG
        for elem in diff3:
            equations.append(sp.re(elem))
            equations.append(sp.im(elem))
        latex_parts.append("(AG)^* = AG")

    if 4 in selected_axioms:
        GA = G @ A
        diff4 = GA.H - GA
        for elem in diff4:
            equations.append(sp.re(elem))
            equations.append(sp.im(elem))
        latex_parts.append("(GA)^* = GA")
    latex_formula = r"\begin{cases} " + r" \\ ".join(latex_parts) + r" \end{cases}"

    try:
        solution = sp.solve(equations, vars_list)

    except Exception as e:
        return None, f"Błąd: {e}", 0
    
    G_final = G.subs(solution)
    return G_final, latex_formula, A.rank()

def calculate_bases(A):
    m, n = A.shape

    try:
        A_rref, pivot_cols = A.rref()
        r = len(pivot_cols)
        if r > 0:
           
            G = A.extract(list(range(m)), list(pivot_cols))
            H = A_rref.extract(list(range(r)), list(range(n)))
            
            part_H = H.H @ (H @ H.H).inv()
            part_G = (G.H @ G).inv() @ G.H
            
            A_plus = part_H @ part_G
        else: 
            A_plus = sp.zeros(n, m)
            part_H = sp.zeros(n, 0)
            part_G = sp.zeros(0, m)

    
        Ker_A = sp.Matrix.hstack(*A.nullspace()) if A.nullspace() else sp.Matrix(n, 0, [])
        Ker_AH = sp.Matrix.hstack(*A.H.nullspace()) if A.H.nullspace() else sp.Matrix(m, 0, [])
        
        
        return A_plus, Ker_A, Ker_AH, r, part_G, part_H
        
    except: return None, None, None, 0, None, None

def generate_parametric(A, selected_axioms):
    m, n = A.shape
    try:
        A_plus, Ker_A, Ker_AH, r, part_G, part_H = calculate_bases(A)
    except: return None, "Błąd baz", 0, ""
    
    dim_ker_A = n - r    
    dim_ker_AH = m - r   
    
    def get_params(rows, cols, name='Y'):
        if rows == 0 or cols == 0: 
            return sp.Matrix(rows, cols, [])
        syms = [sp.Symbol(f'{name}_{i+1}_{j+1}', complex=True) for i in range(rows) for j in range(cols)]
        return sp.Matrix(rows, cols, syms)

    desc = []
    

    if selected_axioms == {1, 2}:
        
        Y = get_params(dim_ker_A, r, 'Y')
        Z = get_params(r, dim_ker_AH, 'Z')
        
        G = (part_H + Ker_A @ Y) @ (part_G + Z @ Ker_AH.H)
        
        form = r"A\{1, 2\} = \left\{ (H^+ + F Y) (G^+ + Z K) \ : \ Y \in \mathbb{C}^{" + f"{dim_ker_A} \\times {r}" + r"}, \ Z \in \mathbb{C}^{" + f"{r} \\times {dim_ker_AH}" + r"} \right\}"
        desc.append(r"$F$ - baza $\mathrm{Ker}(A)$")
        desc.append( r"$K^*$ - baza $\mathrm{Ker}(A^*)$") 
        desc.append(r"$G^+$,$H^+$ - z rozkładu na macierze pełnego rzędu $A=GH$")

    elif selected_axioms == {1}:

        Y = get_params(dim_ker_A, m, 'Y')
        Z = get_params(n, dim_ker_AH, 'Z')
        
        G = A_plus + Ker_A @ Y + Z @ Ker_AH.H
        

        form = r"A\{1\} = \left\{ A^+ + F Y + Z K \ : \ Y \in \mathbb{C}^{" + f"{dim_ker_A}\\times{m}" + r"}, \ Z \in \mathbb{C}^{" + f"{n}\\times{dim_ker_AH}" + r"} \right\}"
        desc.append(r"$F$ - baza $\mathrm{Ker}(A)$")
        desc.append( r"$K^*$ - baza $\mathrm{Ker}(A^*)$") 
        

    elif selected_axioms == {1, 3}:
        Y = get_params(dim_ker_A, m, 'Y')
        G = A_plus + Ker_A @ Y
        form = r"A\{1, 3\} = \left\{ A^+ + F Y \ : \ Y \in \mathbb{C}^{" + f"{dim_ker_A}\\times{m}" + r"} \right\}"
        desc.append(r"$F$ - baza $\mathrm{Ker}(A)$")

    elif selected_axioms == {1, 4}:
        Y = get_params(n, dim_ker_AH, 'Y')
        G = A_plus + Y @ Ker_AH.H
        form = r"A\{1, 4\} = \left\{ A^+ + Y K \ : \ Y \in \mathbb{C}^{" + f"{n}\\times{dim_ker_AH}" + r"} \right\}"
        desc.append( r"$K^*$ - baza $\mathrm{Ker}(A^*)$") 
    elif selected_axioms == {1, 3, 4}:
        Y = get_params(dim_ker_A, dim_ker_AH, 'Y')
        G = A_plus + Ker_A @ Y @ Ker_AH.H
        form = r"A\{1, 3, 4\} = \left\{ A^+ + F Y K \ : \ Y \in \mathbb{C}^{" + f"{dim_ker_A}\\times{dim_ker_AH}" + r"} \right\}"
        desc.append(r"$F$ - baza $\mathrm{Ker}(A)$")
        desc.append( r"$K^*$ - baza $\mathrm{Ker}(A^*)$")  
    elif selected_axioms == {1, 2, 3}: 
        Y = get_params(dim_ker_A, n, 'Y')
        G = A_plus + Ker_A @ Y @ A_plus
        form = r"A\{1, 2, 3\} = \left\{ A^+ + F Y A^+ \ : \ Y \in \mathbb{C}^{" + f"{dim_ker_A}\\times{n}" + r"} \right\}"
        desc.append(r"$F$ - baza $\mathrm{Ker}(A)$") 
    elif selected_axioms == {1, 2, 4}:
        Y = get_params(m, dim_ker_AH, 'Y')
        G = A_plus + A_plus @ Y @ Ker_AH.H
        form = r"A\{1, 2, 4\} = \left\{ A^+ + A^+ Y K \ : \ Y \in \mathbb{C}^{" + f"{m}\\times{dim_ker_AH}" + r"} \right\}"
        desc.append(r"$K^*$ - baza $\mathrm{Ker}(A^*)$") 
    elif selected_axioms in [{1, 2, 3, 4}]:
        G = A_plus
        form = r"A\{1, 2, 3, 4\} = \left\{ A^+ \right\}"
        desc = [] 

    else:
        return None, "Nieobsługiwany zbiór.", 0, ""
    
    if not desc: legend_text = ""
    else: legend_text = "\n".join([f"* {d}" for d in desc])
        
    return G, form, r, legend_text

def rand_complex(rows, cols, real_only=False):
    rng = np.random.default_rng()
    if rows == 0 or cols == 0: 
        dtype = float if real_only else complex
        return np.zeros((rows, cols), dtype=dtype)
    
    real_part = rng.standard_normal((rows, cols))
    
    if real_only:
        return real_part
    else:
        return real_part + 1j * rng.standard_normal((rows, cols))

def generate_idempotent(n, rank, hermitian=False, real_only=False):
    if n == 0: 
        return np.zeros((0,0), dtype=float if real_only else complex)
    
    dt = float if real_only else complex
    
    D = np.zeros((n, n), dtype=dt)
    for i in range(rank):
        D[i, i] = 1.0
        
    if hermitian:
        H = rand_complex(n, n, real_only=real_only)
        U, S, Vh = np.linalg.svd(H) 
        return U @ D @ U.conj().T
    else:
        while True:
            T = rand_complex(n, n, real_only=real_only)
            if abs(np.linalg.det(T)) > 1e-3: 
                break
        return T @ D @ np.linalg.inv(T)

def generate_numerical_svd(A_sympy, selected_axioms):

    rng = np.random.default_rng()

    try:
        A_np = np.array(A_sympy.evalf().tolist()).astype(complex)
    except Exception as e:
        return None, f"Błąd NumPy: {e}", 0, ""

    is_real_matrix = np.allclose(np.imag(A_np), 0, atol=1e-9)

    m, n = A_np.shape
    U, s_vals, Vh = np.linalg.svd(A_np)
    V = Vh.conj().T
    
    tol = 1e-9 * np.max(s_vals) if len(s_vals) > 0 else 0
    r = np.sum(s_vals > tol)
    
    Sigma_r = np.diag(s_vals[:r])
    Sigma_r_inv = np.diag(1.0 / s_vals[:r])
    

    G11 = Sigma_r_inv
    L_block = rand_complex(r, m - r, real_only=is_real_matrix)
    M_block = rand_complex(n - r, r, real_only=is_real_matrix)
    N_block = rand_complex(n - r, m - r, real_only=is_real_matrix)

    latex_eq = ""
    desc = [] 

    if 1 in selected_axioms:
        if selected_axioms == {1}:
            latex_eq = r"G = V \begin{bmatrix} \Sigma_r^{-1} & L \\ M & N \end{bmatrix} U^*"
            desc.append(r"*  $L, M, N$ - macierze dowolne")
        elif selected_axioms == {1, 2}:
            N_block = M_block @ Sigma_r @ L_block
            latex_eq = r"G = V \begin{bmatrix} \Sigma_r^{-1} & L \\ M & M\Sigma_r L \end{bmatrix} U^*"
            desc.append(r"*  $L, M$ - macierze dowolne")
        elif selected_axioms == {1, 3}:
            L_block = np.zeros((r, m - r), dtype=complex)
            latex_eq = r"G = V \begin{bmatrix} \Sigma_r^{-1} & \mathbb{O} \\ M & N \end{bmatrix} U^*"
            desc.append(r"* W$M, N$ - macierze dowolne")
        elif selected_axioms == {1, 4}:
            M_block = np.zeros((n - r, r), dtype=complex)
            latex_eq = r"G = V \begin{bmatrix} \Sigma_r^{-1} & L \\ \mathbb{O} & N \end{bmatrix} U^*"
            desc.append(r"* $L, N$ - macierze dowolne")
        elif selected_axioms == {1, 2, 3}:
            L_block = np.zeros((r, m - r), dtype=complex); N_block = np.zeros((n - r, m - r), dtype=complex)
            latex_eq = r"G = V \begin{bmatrix} \Sigma_r^{-1} & \mathbb{O} \\ M & \mathbb{O} \end{bmatrix} U^*"
            desc.append(r"*  $M$ - macierz dowolna")
        elif selected_axioms == {1, 2, 4}:
            M_block = np.zeros((n - r, r), dtype=complex); N_block = np.zeros((n - r, m - r), dtype=complex)
            latex_eq = r"G = V \begin{bmatrix} \Sigma_r^{-1} & L \\ \mathbb{O} & \mathbb{O} \end{bmatrix} U^*"
            desc.append(r"* $L$ - macierz dowolna")
        elif selected_axioms == {1, 3, 4}:
            L_block = np.zeros((r, m - r), dtype=complex); M_block = np.zeros((n - r, r), dtype=complex)
            latex_eq = r"G = V \begin{bmatrix} \Sigma_r^{-1} & \mathbb{O} \\ \mathbb{O} & N \end{bmatrix} U^*"
            desc.append(r"* $N$ - macierz dowolna")
        elif selected_axioms == {1, 2, 3, 4}:
            L_block = np.zeros((r, m - r), dtype=complex); M_block = np.zeros((n - r, r), dtype=complex); N_block = np.zeros((n - r, m - r), dtype=complex)
            latex_eq = r"G = V \begin{bmatrix} \Sigma_r^{-1} & \mathbb{O} \\ \mathbb{O} & \mathbb{O} \end{bmatrix} U^*"

    elif selected_axioms == {2}:
        s = rng.integers(0, r + 1)
        J = generate_idempotent(r, s, hermitian=False, real_only=is_real_matrix)
        G11 = J @ Sigma_r_inv
        
        L_raw = rand_complex(r, m - r, real_only=is_real_matrix) 
        M_raw = rand_complex(n - r, r, real_only=is_real_matrix)
        L_block = J @ L_raw
        P_row = Sigma_r @ J @ Sigma_r_inv
        M_block = M_raw @ P_row
        N_block = M_block @ Sigma_r @ L_block
        
        latex_eq = r"G = V \begin{bmatrix} G_{11} & L \\ M & M \Sigma_r L \end{bmatrix} U^*"
        
        desc.append(r"Warunki: ")
        desc.append(r"* $G_{11} \Sigma_r G_{11} = G_{11} $ ")
        desc.append(r"*  $G_{11} \Sigma_r L = L$")
        desc.append(r"*  $M \Sigma_r G_{11} = M$")

    elif selected_axioms == {2, 3}:
        s = rng.integers(0, r + 1)
        K = generate_idempotent(r, s, hermitian=True, real_only=is_real_matrix) 
        G11 = Sigma_r_inv @ K
        L_block = np.zeros((r, m - r), dtype=complex); N_block = np.zeros((n - r, m - r), dtype=complex)
        M_block = rand_complex(n - r, r, real_only=is_real_matrix) @ K
        
        latex_eq = r"G = V \begin{bmatrix} G_{11} & \mathbb{O} \\ M & \mathbb{O} \end{bmatrix} U^*"
        desc.append(r"Warunki:")
        desc.append(r"* $G_{11} \Sigma_r G_{11} = G_{11} $ ")
        desc.append(r"* $M \Sigma_r G_{11} = M$ ")
        desc.append(r"* $\Sigma_r G_{11} = G_{11}^*\Sigma_r$")
        desc.append(r"*  $M$ - macierz dowolna")

    elif selected_axioms == {2, 4}:
        s = rng.integers(0, r + 1)
        K = generate_idempotent(r, s, hermitian=True, real_only=is_real_matrix)
        G11 = K @ Sigma_r_inv
        M_block = np.zeros((n - r, r), dtype=complex); N_block = np.zeros((n - r, m - r), dtype=complex)
    
        L_block = K @ rand_complex(r, m - r, real_only=is_real_matrix)
        
        latex_eq = r"G = V \begin{bmatrix} G_{11} & L \\ \mathbb{O} & \mathbb{O} \end{bmatrix} U^*"
        desc.append(r"Warunki:")
        desc.append(r"* $G_{11} \Sigma_r G_{11} = G_{11} $ ")
        desc.append(r"* $G_{11} \Sigma_r L = L$")
        desc.append(r"* $G_{11} \Sigma_r  = \Sigma_r G_{11}^*$")
        desc.append(r"*  $L$ - macierz dowolna")

    elif selected_axioms == {2, 3, 4}:
        dt = float if is_real_matrix else complex 
        
        mask = rng.integers(0, 2, r)
        G11 = np.diag(mask * (1.0 / s_vals[:r])).astype(dt)
        
        L_block = np.zeros((r, m - r), dtype=dt)
        M_block = np.zeros((n - r, r), dtype=dt)
        N_block = np.zeros((n - r, m - r), dtype=dt)
        
        latex_eq = r"G = V \begin{bmatrix} G_{11} & \mathbb{O} \\ \mathbb{O} & \mathbb{O} \end{bmatrix} U^*"
        desc.append(r"Warunki:")
        desc.append(r"* $G_{11} \Sigma_r G_{11} = G_{11} $ ")
        desc.append(r"* $\Sigma_r G_{11} = G_{11}^*\Sigma_r$")
        desc.append(r"* $G_{11} \Sigma_r  = \Sigma_r G_{11}^*$")

    elif selected_axioms == {3}:
        X = rand_complex(r, r, real_only=is_real_matrix)
        U_r, _, _ = np.linalg.svd(X)
        diag_values = rng.standard_normal(r)
        D = np.diag(diag_values)
        H_rand = U_r @ D @ U_r.conj().T
        G11 = Sigma_r_inv @ H_rand; L_block = np.zeros((r, m - r), dtype=complex)
        latex_eq = r"G = V \begin{bmatrix} G_{11} & \mathbb{O} \\ M & N \end{bmatrix} U^*"
        desc.append(r"Warunki:")
        desc.append(r"*  $\Sigma_r G_{11} = (\Sigma_r G_{11})^*$")
        desc.append(r"* $M, N$ - macierze dowolne")

    elif selected_axioms == {4}:
        X = rand_complex(r, r, real_only=is_real_matrix)
        U_r, _, _ = np.linalg.svd(X)
        diag_values = rng.standard_normal(r)
        D = np.diag(diag_values)
        H_rand = U_r @ D @ U_r.conj().T
        G11 = H_rand @ Sigma_r_inv; M_block = np.zeros((n - r, r), dtype=complex)
        latex_eq = r"G = V \begin{bmatrix} G_{11} & L \\ \mathbb{O} & N \end{bmatrix} U^*"
        desc.append(r"Warunki :")
        desc.append(r"*  $G_{11} \Sigma_r = (G_{11} \Sigma_r)^*$")
        desc.append(r"*  $L, N$ - macierze dowolne")

    elif selected_axioms == {3, 4}:
        L_block = np.zeros((r, m - r), dtype=complex); M_block = np.zeros((n - r, r), dtype=complex)
        diag_values = rng.standard_normal(r)
        G11 = np.diag(diag_values)
        latex_eq = r"G = V \begin{bmatrix} G_{11} & \mathbb{O} \\ \mathbb{O} & N \end{bmatrix} U^*"
        desc.append(r" * $N$ - macierz dowolna")
    
    else:
        G11 = np.zeros((r, r), dtype=complex); L_block = np.zeros((r, m - r), dtype=complex)
        M_block = np.zeros((n - r, r), dtype=complex); N_block = np.zeros((n - r, m - r), dtype=complex)
        latex_eq = r"G = \mathbb{O}"

    top_row = np.hstack([G11, L_block])
    bottom_row = np.hstack([M_block, N_block])
    if bottom_row.shape[0] == 0: G_tilde = top_row
    elif top_row.shape[1] == 0: G_tilde = bottom_row 
    else: G_tilde = np.vstack([top_row, bottom_row])

    G_final = V @ G_tilde @ U.conj().T
    if not desc: legend_text = ""
    else: legend_text = "\n".join(desc)
    
    return G_final, latex_eq, r, legend_text

all_sets_options = {
    "{1}": {1}, "{2}": {2}, "{3}": {3}, "{4}": {4},
    "{1, 2}": {1, 2}, "{1, 3}": {1, 3}, "{1, 4}": {1, 4}, 
    "{2, 3}": {2, 3}, "{2, 4}": {2, 4}, "{3, 4}": {3, 4},
    "{1, 3, 4}": {1, 3, 4}, "{1, 2, 3}": {1, 2, 3}, 
    "{1, 2, 4}": {1, 2, 4}, "{2, 3, 4}": {2, 3, 4}, 
    "{1, 2, 3, 4}": {1, 2, 3, 4}
}
symbolic_keys = [
    "{1}", "{3}", "{4}", "{1, 2}", "{1, 3}", "{1, 4}", 
    "{3, 4}", "{1, 3, 4}", "{1, 2, 3}", "{1, 2, 4}", "{1, 2, 3, 4}"
]

calc_mode = st.radio(
    "Tryb obliczeń:",
    ("Symboliczny", "Numeryczny"),
    horizontal=True,
    help="Symboliczny podaje wzór ogólny. Numeryczny generuje przykład losowy."
)

st.title("Generator Uogólnionych Odwrotności")

with st.sidebar:
    st.header("Ustawienia")

    st.divider()
    m_inp = st.number_input("Wiersze (m)", 1, value=2) 
    n_inp = st.number_input("Kolumny (n)", 1, value=3)
    st.divider()
    
    if calc_mode == "Symboliczny": available_options = symbolic_keys
    else: available_options = list(all_sets_options.keys())
    sel_key = st.selectbox("Wybór Zbioru:", available_options, index=0)

    axioms = all_sets_options[sel_key] 
    st.divider()
    with st.expander("Aksjomaty Moore'a-Penrose'a"):
        st.markdown(r"""
        **Warunki:**
        1. $A G A = A$
        2. $G A G = G$
        3. $(A G)^* = A G$
        4. $(G A)^* = G A$
                    
        **Oznaczenia:**
        * $A$ - dana macierz
        * $G$ - szukana macierz 
        """)
st.subheader("Macierz A")
st.caption("Wprowadź wartości dla macierzy w tabeli. Możesz używać liczb zespolonych (np. `2+3i` lub `4j`).")


if 'matrix_df' not in st.session_state or st.session_state.matrix_df.shape != (m_inp, n_inp):
    data = np.zeros((m_inp, n_inp), dtype=str)

    if m_inp >= 2 and n_inp >= 3:
        data[0,0]="1"; data[0,1]="2"; data[0,2]="0"
        data[1,0]="3"; data[1,1]="6"; data[1,2]="0"
    

    rows_labels = range(1, m_inp + 1)
    cols_labels = range(1, n_inp + 1)
    
    st.session_state.matrix_df = pd.DataFrame(data, index=rows_labels, columns=cols_labels)


edited_df = st.data_editor(
    st.session_state.matrix_df, 
    key="editor",
    width='content' 
)

if st.button("Oblicz", type="primary"):
    st.session_state.run_calc = True


if 'run_calc' in st.session_state and st.session_state.run_calc:
    A_matrix, error = parse_dataframe_to_matrix(edited_df)
    
    if error:
        st.error(error) 
        st.session_state.run_calc = False
    else:
        st.divider()
        st.write("Wczytana macierz:")
        show_matrix(A_matrix)

        if calc_mode == "Symboliczny":
            with st.spinner("Generowanie symboliczne..."):
                if axioms in [{3}, {4}, {3, 4}]:
                    G_final, latex_formula, rank = solve_equations(A_matrix, axioms)
                    legend_text = "" 
                else: 
                    G_final, latex_formula, rank, legend_text = generate_parametric(A_matrix, axioms)

                if G_final is None:
                    st.error(latex_formula) 
                else:
                    X_raw = G_final
                    pretty_matrix = prettify_result(X_raw) 
                    
                    st.subheader(f"Wynik symboliczny dla A{sel_key}")
                    st.markdown(f"**Rząd A: r = {rank}**")
                    st.latex(latex_formula)
                    

                    if legend_text:
                        st.caption(legend_text)

                    
                    show_matrix(pretty_matrix)
                    st.divider()

                    st.subheader("Weryfikacja aksjomatów (Symboliczna)")
                    ver_results = verify_axioms(A_matrix, X_raw)
                    
                    cols_ver = st.columns(4)

                    for idx, ax in enumerate([1, 2, 3, 4]):
                        is_passed = ver_results[ax]
                        is_required = ax in axioms

                        icon = "🗸" if is_passed else "🗙"
                        text = "WYMAGANY" if is_required else "DODATKOWY"
                        color_stat = "off" if not is_required else "normal"

                        cols_ver[idx].metric(f"Aksjomat {ax}", icon, text, delta_color=color_stat)

                    free_params = sorted(list(pretty_matrix.free_symbols), key=lambda s: s.name)

                    if len(free_params) > 0:
                        st.divider()

                        st.subheader(
                        f"Podstawianie wartości ({len(free_params)} zm.)", 
                        help="Wprowadź wartości dla parametrów swobodnych, aby otrzymać konkretny przykład macierzy G."
                        )

                        user_subs = {}

                        input_cols = st.columns(min(3, len(free_params)))

                        for i, param in enumerate(free_params):
                            with input_cols[i % 3]:
                                label_latex = f"${sp.latex(param)}$"
                                user_input = st.text_input(label_latex, value="0", key=f"input_{param}")

                                try:
                                    clean_input = clean_input_string(user_input)
                                    math_obj = sp.sympify(clean_input, locals={'I': sp.I})
                                    converted_val = sp.nsimplify(math_obj, tolerance=1e-10, rational=True)
                                    user_subs[param] = converted_val
                                except: 
                                    pass

                        if st.button("Przelicz podstawienie"):  
                            numeric_result = pretty_matrix.subs(user_subs)

                            st.markdown("**Wynik liczbowy (z podstawienia):**")
                            show_matrix(sp.simplify(numeric_result))


                            with st.expander("Sprawdź aksjomaty dla podstawienia"):
                                numeric_verification = verify_axioms(A_matrix, numeric_result)
                                cols_num = st.columns(4)

                                for param_idx, ax_num in enumerate([1, 2, 3, 4]):
                                    is_ok = numeric_verification[ax_num]
                                    icon = "🗸" if is_ok else "🗙"
                                    color = "green" if is_ok else "red"

                                    cols_num[param_idx].markdown(f"**Aksjomat {ax_num}:** :{color}[{icon}]")

                    else: st.info("Wynik jest unikalny (brak parametrów).")

        else: 

            with st.spinner("Generowanie numeryczne..."):

                G_num, explanation_latex, rank_num, legend_text = generate_numerical_svd(A_matrix, axioms)
                
                if G_num is None:
                    st.error(explanation_latex)
                else:
                    st.subheader(f"Przykładowy wynik numeryczny dla A{sel_key}")
                    st.markdown(f"**Rząd A: r = {rank_num}**")
                    
                    if explanation_latex:
                        st.latex(explanation_latex)
                        

                        if legend_text:
                            st.caption(legend_text)
                    
                    st.write("Wygenerowana macierz G:")
                    G_smart_sympy = numpy_to_smart_sympy_matrix(G_num)
                    show_matrix(G_smart_sympy)

                    st.divider()

                    st.subheader("Weryfikacja aksjomatów")
                    ver_results_num = verify_axioms(A_matrix, G_num)

                    cols_ver_num = st.columns(4)

                    for idx, ax in enumerate([1, 2, 3, 4]):
                        is_passed = ver_results_num[ax]
                        is_required = ax in axioms

                        icon = "🗸" if is_passed else "🗙"
                        text = "WYMAGANY" if is_required else "DODATKOWY"
                        color_stat = "off" if not is_required else "normal"

                        cols_ver_num[idx].metric(f"Aksjomat {ax}", icon, text, delta_color=color_stat)
                    
                    if st.button("Generuj inny przykład"):
                        st.rerun()