import streamlit as st
import numpy as np
import pandas as pd

# --- DISPLAY HELPER FUNCTION ---
def display_matrix(matrix, title, alt_names, crit_names=None):
    """Display matrices in a structured table"""
    df = pd.DataFrame(matrix, index=alt_names, columns=crit_names if crit_names else alt_names)
    with st.expander(title, expanded=False):
        st.dataframe(df.style.format("{:.3f}"), use_container_width=True)

# --- MATRIX OPERATIONS ---
def normalize_matrix(matrix):
    """Step 1: Euclidean normalization"""
    return matrix / np.sqrt(np.sum(matrix**2, axis=0))

def apply_weights(normalized_matrix, weights):
    """Step 2: Apply criteria weights"""
    return normalized_matrix * weights

def calculate_concordance(weighted_matrix, weights):
    """Step 3: Compute Concordance Matrix"""
    m = weighted_matrix.shape[0]
    C = np.zeros((m, m))
    
    for k in range(m):
        for l in range(m):
            if k != l:
                concordant = weighted_matrix[k] >= weighted_matrix[l]
                C[k, l] = weights[concordant].sum()
    np.fill_diagonal(C, np.nan)
    return C

def calculate_discordance(weighted_matrix):
    """Step 4: Compute Discordance Matrix"""
    m = weighted_matrix.shape[0]
    D = np.zeros((m, m))
    
    for k in range(m):
        for l in range(m):
            if k != l:
                discordant = weighted_matrix[k] < weighted_matrix[l]
                if np.any(discordant):
                    numerator = np.max(weighted_matrix[l][discordant] - weighted_matrix[k][discordant])
                    denominator = np.max(np.abs(weighted_matrix[k] - weighted_matrix[l]))
                    D[k, l] = numerator / denominator if denominator != 0 else 0
    np.fill_diagonal(D, np.nan)
    return D

def calculate_threshold(matrix):
    """Compute threshold for Concordance/Discordance"""
    m = matrix.shape[0]
    total = np.nansum(matrix)
    return total / (m * (m - 1))

def rank_alternatives(scores, tie_mode):
    """Ranks alternatives, supports tie-breaking mode"""
    sorted_indices = np.argsort(scores)[::-1]  
    ranks = np.zeros_like(scores, dtype=int)

    if tie_mode == "Standard Ranking (1, 2, 3...)":
        for i, idx in enumerate(sorted_indices):
            ranks[idx] = i + 1
    else:
        unique_scores = np.unique(scores)[::-1]  
        rank_value = 1
        for score in unique_scores:
            tied_indices = np.where(scores == score)[0]
            for idx in tied_indices:
                ranks[idx] = rank_value
            rank_value += len(tied_indices)

    return ranks

# --- ELECTRE IMPLEMENTATION ---
def electre_iv(matrix, weights, alt_names, tie_mode):
    """Main ELECTRE function with tie-breaking"""
    R = normalize_matrix(matrix)
    display_matrix(R, "1. Normalized Matrix", alt_names, [f"C{i+1}" for i in range(matrix.shape[1])])
    
    V = apply_weights(R, weights)
    display_matrix(V, "2. Weighted Matrix", alt_names, [f"C{i+1}" for i in range(matrix.shape[1])])
    
    C = calculate_concordance(V, weights)
    display_matrix(C, "3. Concordance Matrix", alt_names)
    
    D = calculate_discordance(V)
    display_matrix(D, "4. Discordance Matrix", alt_names)
    
    c_threshold = calculate_threshold(C)
    F = (C >= c_threshold).astype(float)
    F[np.isnan(F)] = np.nan
    display_matrix(F, "5a. Concordance Dominance Matrix", alt_names)
    
    d_threshold = calculate_threshold(D)
    G = (D >= d_threshold).astype(float)
    G[np.isnan(G)] = np.nan
    display_matrix(G, "5b. Discordance Dominance Matrix", alt_names)
    
    E = F * G
    display_matrix(E, "6. Aggregate Dominance Matrix", alt_names)
    
    scores = np.nansum(E, axis=1)
    ranks = rank_alternatives(scores, tie_mode)
    
    return scores, ranks

# --- STREAMLIT APP ---
def main():
    st.title("⚡ ELECTRE Decision Support System")
    st.markdown("**Multi-Criteria Decision Analysis (MCDA) tool for ranking alternatives based on multiple criteria.**")

    st.sidebar.header("🔧 Input Setup")
    num_alts = st.sidebar.slider("Number of Alternatives", 1, 10, 4)
    num_crits = st.sidebar.slider("Number of Criteria", 1, 10, 4)

    alt_names = [f"Alternative {i+1}" for i in range(num_alts)]

    st.subheader("📊 Enter Decision Matrix")
    matrix = np.zeros((num_alts, num_crits))

    for i in range(num_alts):
        cols = st.columns(num_crits)
        for j in range(num_crits):
            matrix[i, j] = cols[j].number_input(f"{alt_names[i]} - C{j+1}", min_value=0.0, value=1.0, step=0.1)

    st.subheader("⚖️ Set Criteria Weights")
    weights = np.zeros(num_crits)

    for j in range(num_crits):
        weights[j] = st.slider(f"Weight for C{j+1}", 1, 10, 5)

    tie_mode = st.selectbox(
        "📌 Choose Ranking Mode:",
        ["Standard Ranking (1, 2, 3...)", "Tie-Friendly Ranking (1, 1, 2, 3...)"],
        index=0
    )

    if st.button("🚀 Run ELECTRE"):
        scores, ranks = electre_iv(matrix, weights, alt_names, tie_mode)

        st.subheader("🏆 Final Rankings")
        score_df = pd.DataFrame({
            "Alternative": alt_names,
            "Score": scores,
            "Rank": ranks
        }).set_index("Alternative")

        st.dataframe(score_df.style.highlight_max(axis=0))

        best_alts = [alt_names[i] for i in np.where(ranks == 1)[0]]
        best_result = " 🎖️ Best Alternative: " + ", ".join(best_alts)
        st.success(best_result)

if __name__ == "__main__":
    main()
