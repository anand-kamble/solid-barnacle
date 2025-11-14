#!/usr/bin/env python3
"""
Fix encoding issues in v2_mermaid.md
Replace Unicode math symbols with ASCII equivalents
"""

def fix_encoding(text):
    """Apply careful, context-aware replacements"""

    # Fix Poincaré (corrupted as Poincar�)
    text = text.replace('Poincar� Ball', 'Poincare Ball')

    # Fix specific corrupted patterns in formulas
    # Line 10: "Child \b Cone" -> "Child in Cone" (\b is backspace char)
    text = text.replace('Child \x08 Cone', 'Child in Cone')
    # Line 11: "Non-descendant \t Cone" -> "Non-descendant not in Cone" (\t is tab)
    text = text.replace('Non-descendant \x09 Cone', 'Non-descendant not in Cone')

    # Fix lambda symbols in formulas
    text = text.replace('�_neg�L_neg', 'lambda_neg*L_neg')
    text = text.replace('�_reg�L_reg', 'lambda_reg*L_reg')
    text = text.replace('�_div�L_div', 'lambda_div*L_div')
    text = text.replace('�_div', 'lambda_div')

    # Fix set membership in context
    text = text.replace('E \x08 L^d', 'E in L^d')
    text = text.replace('h_text \x08 R^1536', 'h_text in R^1536')
    text = text.replace('h_context \x08 R^d_model', 'h_context in R^d_model')
    text = text.replace('H_patterns \x08 R^K�d_pattern', 'H_patterns in R^K*d_pattern')
    text = text.replace('h_line \x08 R^d_model', 'h_line in R^d_model')
    text = text.replace('q_e \x08 R^d_emb', 'q_e in R^d_emb')

    # Fix angle brackets (Lorentz inner product notation)
    text = text.replace('�x,x�_L', '<x,x>_L')
    text = text.replace('�q,e_i�_L', '<q,e_i>_L')

    # Fix mathematical formulas with theta and epsilon
    text = text.replace('� x = arcsin ��1-\x16x\x16�/\x16x\x16+�', 'theta_x = arcsin(sqrt(1-||x||^2)/(||x||+epsilon))')
    text = text.replace('� x,y', 'theta(x,y)')
    text = text.replace('� x', 'theta_x')

    # Fix ReLU formula
    text = text.replace('E_cone = ReLU �-�', 'E_cone = ReLU(theta-alpha)')
    text = text.replace('E_cone x,y = ReLU � x,y - � x', 'E_cone(x,y) = ReLU(theta(x,y) - theta_x)')

    # Fix threshold symbols
    text = text.replace('prob e �_exact', 'prob >= tau_exact')
    text = text.replace('prob e �_parent', 'prob >= tau_parent')
    text = text.replace('prob e �_grandparent', 'prob >= tau_grandparent')
    text = text.replace('prob < �_min', 'prob < tau_min')
    text = text.replace('p_max e �_exact', 'p_max >= tau_exact')
    text = text.replace('p_max e �_parent', 'p_max >= tau_parent')
    text = text.replace('p_max e �_grandparent', 'p_max >= tau_grandparent')
    text = text.replace('p_max e �_min', 'p_max >= tau_min')

    # Fix summation symbols
    text = text.replace('� pairs across', 'for all pairs across')
    text = text.replace('N_accounts � N_accounts', 'N_accounts x N_accounts')
    text = text.replace('� p_di', 'sum p_di')
    text = text.replace('� p_cj', 'sum p_cj')
    text = text.replace('� p_same_side', 'sum p_same_side')
    text = text.replace('�_debit', 'sigma_debit')
    text = text.replace('�_credit', 'sigma_credit')
    text = text.replace('�_all_debits', 'sigma_all_debits')
    text = text.replace('�_all_credits', 'sigma_all_credits')
    text = text.replace('� debits', 'sum debits')
    text = text.replace('� credits', 'sum credits')
    text = text.replace('� edges', 'sum over edges')
    text = text.replace('� non-descendants', 'sum over non-descendants')
    text = text.replace('�i<j', 'sum_i<j')
    text = text.replace('� exp s_j', 'sum exp(s_j)')
    text = text.replace('� all components', 'sum all components')

    # Fix arrow symbols (but keep --> for mermaid)
    text = text.replace('Linear � ReLU � Linear', 'Linear -> ReLU -> Linear')
    text = text.replace('Linear � Sigmoid � p', 'Linear -> Sigmoid -> p')
    text = text.replace('Euclidean � Lorentz', 'Euclidean -> Lorentz')
    text = text.replace('Distances � Scores', 'Distances -> Scores')
    text = text.replace('Linear d_model � d_emb', 'Linear(d_model -> d_emb)')

    # Fix multiplication symbols
    text = text.replace('proportion � A', 'proportion * A')
    text = text.replace('x0 = \u001a1+\x16x\x16�', 'x0 = sqrt(1+||x||^2)')

    # Fix cone angles in examples
    text = text.replace('large cone � Assets', 'large cone theta_Assets')
    text = text.replace('medium cone � Cash', 'medium cone theta_Cash')
    text = text.replace('medium cone � AR', 'medium cone theta_AR')
    text = text.replace('small cone � BankA', 'small cone theta_BankA')
    text = text.replace('small cone � BankB', 'small cone theta_BankB')
    text = text.replace('small cone � Cust1', 'small cone theta_Cust1')
    text = text.replace('small cone � Cust2', 'small cone theta_Cust2')

    # Fix cone membership examples
    text = text.replace('y \x08 Cone x', 'y in Cone(x)')
    text = text.replace('Cash \x08 Cone Assets', 'Cash in Cone(Assets)')
    text = text.replace('Bank A \x08 Cone Cash', 'Bank A in Cone(Cash)')
    text = text.replace('Bank A \x08 Cone Assets', 'Bank A in Cone(Assets)')
    text = text.replace('Cust1 \x08 Cone Cash', 'Cust1 in Cone(Cash)')

    # Fix inequality symbols
    text = text.replace(' d � x', ' <= theta_x')
    text = text.replace(' d � parent', ' <= theta_parent')
    text = text.replace('� Assets,Cash d � Assets', 'theta(Assets,Cash) <= theta_Assets')
    text = text.replace('� Cash,BankA d � Cash', 'theta(Cash,BankA) <= theta_Cash')
    text = text.replace('� Assets,BankA d � Assets', 'theta(Assets,BankA) <= theta_Assets')
    text = text.replace('� Cash,Cust1 > � Cash', 'theta(Cash,Cust1) > theta_Cash')

    # Fix angle bracket in cone definition
    text = text.replace('� x,y d � x', 'theta(x,y) <= theta_x')

    # Fix ReLU with margin
    text = text.replace('ReLU margin - �-�', 'ReLU(margin - theta)')

    # Fix checkmarks and crosses
    text = text.replace('\u0013', '✓')
    text = text.replace('\u0017', '✗')

    # Fix critical paths arrows
    text = text.replace('Data � Encode � Retrieve � Attend � Decode � Loss � Update',
                       'Data -> Encode -> Retrieve -> Attend -> Decode -> Loss -> Update')
    text = text.replace('Text � Encode � Retrieve � Attend � Generate K � Rank � Output',
                       'Text -> Encode -> Retrieve -> Attend -> Generate K -> Rank -> Output')
    text = text.replace('Metadata � Parent Embedding � Exponential Map � Verify Cone',
                       'Metadata -> Parent Embedding -> Exponential Map -> Verify Cone')

    # Fix complexity notation
    text = text.replace('O(K�)', 'O(K^2)')

    # Fix inference notation in parent calc
    text = text.replace('� parent, new d � parent', 'theta(parent, new) <= theta_parent')

    return text

# Read, fix, and write
with open('v2_mermaid.md', 'r', encoding='utf-8') as f:
    content = f.read()

fixed_content = fix_encoding(content)

with open('v2_mermaid.md', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("Successfully fixed encoding issues!")
print("Main changes:")
print("- Poincaré -> Poincare")
print("- Unicode math symbols -> ASCII equivalents")
print("- λ -> lambda")
print("- θ -> theta")
print("- ε -> epsilon")
print("- ∑ -> sum")
print("- × -> x or *")
print("- → -> ->")
print("- ≤ -> <=")
print("- ∈ -> in")
