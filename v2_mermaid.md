# Comprehensive Mermaid Diagram for Journal Entry Prediction Architecture (V2)

## Full System Architecture

```mermaid
graph TB
    subgraph "STAGE 1: UNSUPERVISED EMBEDDING PRE-TRAINING"
        A1[Chart of Accounts<br/>Hierarchy Tree] --> A2[Lorentz Model<br/>Optimization]
        A2 --> A3{Compute Cone<br/>Energy in<br/>Poincare Ball}
        A3 --> A4[L_pos: Child in Cone Parent]
        A3 --> A5[L_neg: Non-descendant not in Cone]
        A3 --> A6[L_reg: Boundary Regularization]
        A4 --> A7[L_total = L_pos + lambda_neg*L_neg + lambda_reg*L_reg]
        A5 --> A7
        A6 --> A7
        A7 --> A8[Riemannian SGD/Adam<br/>on Lorentz Manifold]
        A8 --> A9[Fixed Account Embeddings<br/>E  L^d d=32-64<br/><x,x>_L = -1]
        A9 --> A10[Cone Properties:<br/>theta_x = arcsin in  in 1-x in /x+ in <br/>E_cone = ReLU(theta-alpha)]
    end

    subgraph "PATTERN INDEXING OFFLINE"
        B1[Historical Journal<br/>Entries] --> B2[Group by Signature<br/>tuple debit_accts, credit_accts]
        B2 --> B3[For each pattern:<br/>Compute text centroid<br/>Build graph structure]
        B3 --> B4[Pattern Database<br/>signature, graph,<br/>text_centroid, support]
        B4 --> B5[Build FAISS Index<br/>or Cosine Similarity<br/>Matrix]
        B5 --> B6[PatternRetriever<br/>Ready for Queries]
    end

    subgraph "GLOBAL CO-OCCURRENCE GRAPH OFFLINE"
        C1[All Historical<br/>Journal Entries] --> C2[Extract Account Pairs<br/>from each entry]
        C2 --> C3[Count Co-occurrences<br/> in pairs across all entries]
        C3 --> C4[Normalize Weights<br/>by total entries]
        C4 --> C5[Sparse Matrix<br/>N_accounts in N_accounts]
    end

    subgraph "STAGE 2: TASK MODEL TRAINING"
        D1[Training Data:<br/>Text Description +<br/>Ground Truth JE] --> D2[Text Encoder<br/>OpenAI Ada-002<br/>FROZEN]
        D2 --> D3[h_text  R^1536]

        D1 --> D4[Query PatternRetriever<br/>Retrieve K=10 similar<br/>historical patterns]
        B6 -.Offline Index.-> D4

        D4 --> D5[Retrieved Patterns<br/>signature, graph,<br/>text_centroid, support]
        D5 --> D6[Encode Pattern Graphs<br/>debit_emb + credit_emb<br/>weighted by ratios]
        A9 -.Fixed Embeddings.-> D6
        D6 --> D7[H_patterns  R^K in d_pattern]

        D3 --> D8[Cross-Attention<br/>Context Integrator]
        D7 --> D8
        D8 --> D9[h_context  R^d_model<br/>d_model=256]

        D9 --> D10[Multi-Hypothesis<br/>Decoder<br/>K Parallel Heads]
    end

    subgraph "SINGLE HYPOTHESIS DECODER k=1..K"
        E1[h_context] --> E2[Line Count Predictor<br/>Linear in ReLU in Linear]
        E2 --> E3[n_lines_logits]
        E3 --> E4[n_lines = argmax]

        E1 --> E5[Transformer Decoder<br/>3 Layers, 8 Heads]
        E5 --> E6[For i in 1..N-1<br/>Decode Line i]

        E6 --> E7[Account Predictor<br/>Hierarchical]
        E6 --> E8[D/C Predictor<br/>Linear 2 classes]
        E6 --> E9[Proportion Predictor<br/>Linear in Sigmoid in p0,1]

        E7 --> E10[Line i:<br/>account_id, level,<br/>debit_credit, proportion]
        E8 --> E10
        E9 --> E10

        E10 --> E11{i < N-1?}
        E11 -->|Yes| E6
        E11 -->|No| E12[Derive Last Line N]

        E12 --> E13[Predict: account, D/C<br/>Derive: p_last = 1 - in p_same_side]
        E13 --> E14[Complete Journal Entry<br/>N Lines<br/>GUARANTEED BALANCED<br/>per-side in p = 1]
    end

    subgraph "HIERARCHICAL ACCOUNT PREDICTOR"
        F1[h_line  R^d_model] --> F2[Query Projection<br/>Linear d_model in d_emb]
        F2 --> F3[q_e  R^d_emb]
        F3 --> F4[Euclidean in Lorentz<br/>q_L = x0, x1..xd<br/>x0 = 1+x in ]

        F4 --> F5{Search Mode?}
        F5 -->|Full| F6[Compute d_L q_L, E<br/>for all accounts]
        F5 -->|Top-Down| F7[Tree Traversal<br/>Start at root,<br/>pick closest child]

        F6 --> F8[Distances in Scores<br/>s = -d/T]
        F7 --> F8
        A9 -.Fixed E.-> F6
        A9 -.Fixed E.-> F7

        F8 --> F9[Softmax Probabilities]
        F9 --> F10[top_idx, max_prob]

        F10 --> F11[Get Path to Root<br/>node, parent, ..., root]
        F11 --> F12{Confidence<br/>Thresholds}
        F12 -->|prob e in _exact=0.8| F13[Return Exact<br/>account_id, level='exact']
        F12 -->|prob e in _parent=0.65| F14[Return Parent<br/>parent_id, level='parent']
        F12 -->|prob e in _grandparent=0.5| F15[Return Grandparent<br/>gp_id, level='grandparent']
        F12 -->|prob < in _min| F16[Return Ancestor<br/>ancestor_id, level='ancestor_k']
    end

    subgraph "LOSS COMPUTATION WINNER-TAKES-ALL"
        G1[K Hypotheses] --> G2[Compute Loss<br/>for each hypothesis k]
        G2 --> G3[Loss_k =<br/>L_line_count +<br/> in L_account +<br/> in L_dc +<br/> in L_proportion]

        G3 --> G4[L_account hierarchical:<br/>exact=0<br/>parent=0.5<br/>grandparent=0.7<br/>wrong=1.0]

        G3 --> G5[Select Winner<br/>winner_idx = argmin Loss_k]
        G5 --> G6[min_loss = Loss_winner]

        G1 --> G7[Diversity Loss<br/>Repulsion between hypotheses]
        G7 --> G8[L_div = - in i<j similarity hyp_i, hyp_j]

        G6 --> G9[Total Loss =<br/>min_loss + 0.01 in L_div]
        G8 --> G9

        G9 --> G10[Backprop through<br/>Winner Decoder Only]
        G10 --> G11[Optimizer Step<br/>Adam, lr=1e-3<br/>Gradient Clipping]
    end

    subgraph "INFERENCE RUNTIME"
        H1[New Text<br/>Description] --> H2[Text Encoder<br/>FROZEN]
        H2 --> H3[h_text]

        H1 --> H4[Query PatternRetriever<br/>K=10 similar patterns]
        B6 -.Index.-> H4
        H4 --> H5[Retrieved Patterns]

        H5 --> H6[Encode Pattern Graphs]
        A9 -.Embeddings.-> H6
        H6 --> H7[H_patterns]

        H3 --> H8[Cross-Attention]
        H7 --> H8
        H8 --> H9[h_context]

        H9 --> H10[Multi-Hypothesis Decoder<br/>Generate K=5 hypotheses]
        H10 --> H11[Hypothesis 1]
        H10 --> H12[Hypothesis 2]
        H10 --> H13[Hypothesis 3]
        H10 --> H14[Hypothesis K]

        H11 --> H15[Compute Confidence<br/>geometric mean of<br/>line confidences]
        H12 --> H15
        H13 --> H15
        H14 --> H15

        H15 --> H16[Sort by Confidence<br/>Descending]
        H16 --> H17[Top-K Predictions<br/>with Confidence Scores<br/>and Attention Weights]
    end

    subgraph "NEW ACCOUNT HANDLING INDUCTIVE"
        I1[New Account<br/>Metadata] --> I2[Get Parent Embedding<br/>E parent_id]
        A9 -.Embeddings.-> I2
        I2 --> I3[Compute Direction<br/>based on account_type<br/>and depth]
        I3 --> I4[Sample Tangent Vector<br/>in parent's cone]
        I4 --> I5[Exponential Map<br/>new_emb = exp_parent direction]
        I5 --> I6[Verify Cone Membership<br/> in parent, new d in parent]
        I6 --> I7[New Account Embedding<br/>NO TRAINING REQUIRED]
    end

    subgraph "BALANCE GUARANTEE MECHANISM"
        J1[Predict Lines 1..N-1] --> J2[Each line:<br/>account, D/C, proportion p]
        J2 --> J3[Group by D/C side]
        J3 --> J4[Debit Side:<br/>p_d1, p_d2, ..., p_dk]
        J3 --> J5[Credit Side:<br/>p_c1, p_c2, ..., p_cm]

        J4 --> J6[ in _debit = in p_di]
        J5 --> J7[ in _credit = in p_cj]

        J6 --> J8{Last Line D/C?}
        J7 --> J8

        J8 -->|Debit| J9[p_last = max 0, 1 - in _debit]
        J8 -->|Credit| J10[p_last = max 0, 1 - in _credit]

        J9 --> J11[GUARANTEE:<br/> in _all_debits = 1.0<br/> in _all_credits = 1.0]
        J10 --> J11

        J11 --> J12[Convert to amounts:<br/>amount = proportion in A<br/>where A = total per side]
        J12 --> J13[Perfect Balance:<br/> in debits = in credits]
    end

    %% Cross-stage connections
    A9 -.Pre-trained.-> D6
    A9 -.Pre-trained.-> H6
    B6 -.Index.-> D4
    D10 --> E1
    H10 --> H11

    %% Styling
    style A9 fill:#ffcccc,stroke:#ff0000,stroke-width:3px
    style B6 fill:#ccffcc,stroke:#00ff00,stroke-width:3px
    style D9 fill:#ccccff,stroke:#0000ff,stroke-width:3px
    style E14 fill:#ffffcc,stroke:#ffff00,stroke-width:3px
    style G9 fill:#ffccff,stroke:#ff00ff,stroke-width:3px
    style H17 fill:#ccffff,stroke:#00ffff,stroke-width:3px
    style I7 fill:#ffeecc,stroke:#ff8800,stroke-width:3px
    style J13 fill:#ccffee,stroke:#00ff88,stroke-width:3px
```

## Training Pipeline Flow

```mermaid
graph LR
    subgraph "Phase 1: Offline Preparation"
        P1[Chart of<br/>Accounts] --> P2[Train Lorentz<br/>Embeddings<br/>100-1000 epochs]
        P2 --> P3[Save Fixed<br/>Embeddings<br/>E.pkl]

        P4[Historical<br/>JEs] --> P5[Build Pattern<br/>Index]
        P5 --> P6[Save PatternRetriever<br/>patterns.pkl]

        P4 --> P7[Build Co-occurrence<br/>Graph]
        P7 --> P8[Save Sparse Matrix<br/>cooccur.npz]
    end

    subgraph "Phase 2: Model Training"
        T1[Training Set<br/>text + JE] --> T2[Epoch Loop<br/>100 epochs]
        P3 -.Load.-> T2
        P6 -.Load.-> T2

        T2 --> T3[Batch:<br/>encode text,<br/>retrieve patterns]
        T3 --> T4[Forward Pass<br/>K hypotheses]
        T4 --> T5[Compute Loss<br/>winner-takes-all]
        T5 --> T6[Backward Pass<br/>clip gradients]
        T6 --> T7[Optimizer Step<br/>Adam]
        T7 --> T8{Converged?}
        T8 -->|No| T2
        T8 -->|Yes| T9[Save Model<br/>model.pth]
    end

    subgraph "Phase 3: Inference"
        I1[New Text] --> I2[Load Model<br/>+ Embeddings<br/>+ PatternRetriever]
        P3 -.Load.-> I2
        P6 -.Load.-> I2
        T9 -.Load.-> I2
        I2 --> I3[Encode + Retrieve]
        I3 --> I4[Generate K<br/>Hypotheses]
        I4 --> I5[Rank by<br/>Confidence]
        I5 --> I6[Return Top-K<br/>Predictions]
    end

    style P3 fill:#ffcccc
    style P6 fill:#ccffcc
    style T9 fill:#ccccff
    style I6 fill:#ccffff
```

## Data Flow During Inference (Detailed)

```mermaid
sequenceDiagram
    participant User
    participant TextEncoder
    participant PatternRetriever
    participant ContextIntegrator
    participant MultiHypothesisDecoder
    participant HierarchicalAccountPredictor
    participant Output

    User->>TextEncoder: description: "Sale of goods for cash"
    TextEncoder->>TextEncoder: OpenAI Ada-002 encode
    TextEncoder-->>ContextIntegrator: h_text  R^1536

    User->>PatternRetriever: description: "Sale of goods for cash"
    PatternRetriever->>PatternRetriever: Embed query text
    PatternRetriever->>PatternRetriever: FAISS search / cosine similarity
    PatternRetriever-->>ContextIntegrator: K=10 patterns [graph, centroid, support]

    ContextIntegrator->>ContextIntegrator: Encode pattern graphs<br/>using fixed account embeddings
    ContextIntegrator->>ContextIntegrator: Cross-attention (Q=h_text, K=H_patterns)
    ContextIntegrator-->>MultiHypothesisDecoder: h_context  R^256

    loop For k=1 to K=5
        MultiHypothesisDecoder->>MultiHypothesisDecoder: Decoder_k predicts n_lines
        loop For line i=1 to N-1
            MultiHypothesisDecoder->>HierarchicalAccountPredictor: h_line_i
            HierarchicalAccountPredictor->>HierarchicalAccountPredictor: Project to Lorentz space
            HierarchicalAccountPredictor->>HierarchicalAccountPredictor: Compute distances to all accounts
            HierarchicalAccountPredictor->>HierarchicalAccountPredictor: Apply confidence thresholds
            HierarchicalAccountPredictor-->>MultiHypothesisDecoder: account_id, level, confidence
            MultiHypothesisDecoder->>MultiHypothesisDecoder: Predict D/C and proportion
        end
        MultiHypothesisDecoder->>MultiHypothesisDecoder: Derive last line (balance guarantee)
        MultiHypothesisDecoder->>MultiHypothesisDecoder: Compute hypothesis confidence
    end

    MultiHypothesisDecoder-->>Output: K hypotheses with confidences
    Output->>Output: Sort by confidence descending
    Output-->>User: Top-K predictions with<br/>confidence scores + attention weights
```

## Hierarchical Account Prediction Decision Tree

```mermaid
graph TD
    A[Query Embedding q_L<br/>in Lorentz Space] --> B[Compute Distances to<br/>All Account Embeddings]
    B --> C[d_L q, e_i = acosh - in q,e_i in _L]
    C --> D[Convert to Scores<br/>s_i = -d_i / T]
    D --> E[Softmax Probabilities<br/>p_i = exp s_i / in exp s_j]
    E --> F[Get Top Account<br/>idx, p_max]

    F --> G[Get Path to Root<br/>path = node, parent, ..., root]

    G --> H{p_max e in _exact<br/>default: 0.8?}
    H -->|Yes| I[Return: node<br/>Level: EXACT<br/>Confidence: p_max]
    H -->|No| J{p_max e in _parent<br/>default: 0.65?}

    J -->|Yes| K{Is parent<br/>of node?}
    K -->|Yes| L[Return: parent<br/>Level: PARENT<br/>Confidence: p_max]
    K -->|No| J

    J -->|No| M{p_max e in _grandparent<br/>default: 0.5?}
    M -->|Yes| N{Is grandparent<br/>of node?}
    N -->|Yes| O[Return: grandparent<br/>Level: GRANDPARENT<br/>Confidence: p_max]
    N -->|No| M

    M -->|No| P{p_max e in _min<br/>default: 0.3?}
    P -->|Yes| Q[Return: nearest ancestor<br/>Level: ANCESTOR_k<br/>Confidence: p_max]
    P -->|No| R[Return: root account<br/>Level: ROOT<br/>Confidence: p_max]

    style I fill:#00ff00
    style L fill:#88ff88
    style O fill:#ccffcc
    style Q fill:#ffffcc
    style R fill:#ffcccc
```

## Entailment Cone Geometry (Hyperbolic Space)

```mermaid
graph TB
    subgraph "Poincare Ball (for cone computation)"
        P1[Assets<br/>near origin<br/>large cone in Assets]
        P1 --> P2[Cash<br/>medium distance<br/>medium cone in Cash]
        P1 --> P3[Accounts Receivable<br/>medium distance<br/>medium cone in AR]

        P2 --> P4[Cash - Bank A<br/>near boundary<br/>small cone in BankA]
        P2 --> P5[Cash - Bank B<br/>near boundary<br/>small cone in BankB]

        P3 --> P6[AR - Customer 1<br/>near boundary<br/>small cone in Cust1]
        P3 --> P7[AR - Customer 2<br/>near boundary<br/>small cone in Cust2]
    end

    subgraph "Cone Containment Rules"
        C1[theta_x = arcsin in  in 1-x in /x+ in ]
        C2[ in x,y = geodesic angle from x to y]
        C3[y in Cone x in  in x,y d in x]

        C1 --> C3
        C2 --> C3

        C4[E_cone x,y = ReLU in x,y - in x]
        C3 --> C4

        C5[Loss minimizes:<br/>L_pos = in edges E_cone parent,child]
        C6[L_neg = in non-descendants ReLU margin - in - in ]
        C4 --> C5
        C4 --> C6
    end

    subgraph "Example Checks"
        E1[Cash in Cone Assets?<br/> in Assets,Cash d in Assets<br/> TRUE]
        E2[Bank A in Cone Cash?<br/> in Cash,BankA d in Cash<br/> TRUE]
        E3[Bank A in Cone Assets?<br/> in Assets,BankA d in Assets<br/> TRUE transitivity]
        E4[Cust1 in Cone Cash?<br/> in Cash,Cust1 > in Cash<br/> FALSE different branch]
    end

    style P1 fill:#ffeeee
    style P2 fill:#ffcccc
    style P3 fill:#ffcccc
    style P4 fill:#ff9999
    style P5 fill:#ff9999
    style P6 fill:#ff9999
    style P7 fill:#ff9999
    style E1 fill:#ccffcc
    style E2 fill:#ccffcc
    style E3 fill:#ccffcc
    style E4 fill:#ffcccc
```

## Loss Computation Breakdown

```mermaid
graph TB
    subgraph "Hypothesis-Level Loss Single Hypothesis k"
        HL1[Predicted JE] --> HL2[1. Line Count Loss<br/>CE n_lines_logits, n_true]

        HL1 --> HL3[2. Per-Line Losses<br/>for lines 1..N-1]
        HL3 --> HL4[Account Loss hierarchical]
        HL3 --> HL5[D/C Loss<br/>CE dc_logits, dc_true]
        HL3 --> HL6[Proportion Loss<br/>MSE p_pred, p_true]

        HL1 --> HL7[3. Last Line N]
        HL7 --> HL8[Account Loss hierarchical<br/>NO proportion loss]
        HL7 --> HL9[D/C Loss]

        HL2 --> HL10[L_hypothesis_k = in all components]
        HL4 --> HL10
        HL5 --> HL10
        HL6 --> HL10
        HL8 --> HL10
        HL9 --> HL10
    end

    subgraph "Hierarchical Account Loss Detail"
        HAL1{Prediction vs<br/>Ground Truth} -->|Exact Match| HAL2[Loss = 0.0<br/>Perfect prediction]
        HAL1 -->|Parent Match| HAL3[Loss = 0.5<br/>Partial credit 50%]
        HAL1 -->|Grandparent Match| HAL4[Loss = 0.7<br/>Partial credit 30%]
        HAL1 -->|Wrong Branch| HAL5[Loss = 1.0<br/>Full penalty]
    end

    subgraph "Multi-Hypothesis Total Loss"
        MHL1[K Hypotheses<br/>L_1, L_2, ..., L_K] --> MHL2[Winner Selection<br/>winner = argmin L_k]
        MHL2 --> MHL3[min_loss = L_winner]

        MHL1 --> MHL4[Diversity Regularization]
        MHL4 --> MHL5[L_div = - in i<j similarity hyp_i, hyp_j]
        MHL5 --> MHL6[Encourage different predictions]

        MHL3 --> MHL7[L_total = min_loss + lambda_div*L_div<br/> in _div = 0.01]
        MHL6 --> MHL7

        MHL7 --> MHL8[Backprop through<br/>Winner Decoder ONLY]
    end

    style HAL2 fill:#00ff00
    style HAL3 fill:#88ff88
    style HAL4 fill:#ccffcc
    style HAL5 fill:#ffcccc
    style MHL3 fill:#ccccff
    style MHL7 fill:#ffccff
```

## Module Dependencies and Interfaces

```mermaid
graph LR
    subgraph "Embedding Module embeddings/"
        EM1[TextEncoder<br/>OpenAI Ada-002<br/>FROZEN]
        EM2[LorentzEmbedding<br/>Pre-trained<br/>FROZEN]
    end

    subgraph "Retrieval Module retrieval/"
        RM1[PatternRetriever<br/>build index<br/>search query]
        RM2[CoOccurrenceGraph<br/>global graph<br/>N in N sparse]
    end

    subgraph "Core Model core/"
        CM1[InputEncoder<br/>text + pattern graph]
        CM2[ContextIntegrator<br/>cross-attention]
        CM3[MultiHypothesisDecoder<br/>K parallel heads]
        CM4[JournalEntryDecoder<br/>single hypothesis]
        CM5[HierarchicalAccountPredictor<br/>Lorentz distance]
    end

    subgraph "Training training/"
        TR1[TrainingLoop<br/>winner-takes-all]
        TR2[LossComputation<br/>hierarchical + diversity]
    end

    subgraph "Inference inference/"
        IN1[PredictionPipeline<br/>end-to-end]
        IN2[ConfidenceCalibration<br/>score ranking]
    end

    EM1 --> CM1
    EM1 --> RM1
    EM2 --> CM1
    EM2 --> CM5
    EM2 --> IN1

    RM1 --> CM1
    RM1 --> IN1
    RM2 -.Optional.-> IN1

    CM1 --> CM2
    CM2 --> CM3
    CM3 --> CM4
    CM4 --> CM5

    CM3 --> TR2
    TR2 --> TR1

    CM1 --> IN1
    CM2 --> IN1
    CM3 --> IN1
    IN1 --> IN2

    style EM1 fill:#ffeeee
    style EM2 fill:#ffcccc
    style RM1 fill:#eeffee
    style CM3 fill:#eeeeff
    style TR1 fill:#ffeeff
    style IN1 fill:#eeffff
```

## Performance Metrics Dashboard

```mermaid
graph TB
    subgraph "Accuracy Metrics"
        ACC1[Top-1 Exact Account<br/>Target: 70-80%]
        ACC2[Top-1 Parent Account<br/>Target: 82-90%]
        ACC3[Top-3 Hypothesis Coverage<br/>Target: 88-94%]
        ACC4[Top-5 Hypothesis Coverage<br/>Target: 92-96%]
    end

    subgraph "Confidence Metrics"
        CONF1[Expected Calibration Error<br/>Target: <0.08]
        CONF2[Confidence-Accuracy<br/>Correlation<br/>Target: >0.8]
        CONF3[Brier Score<br/>Target: <0.15]
    end

    subgraph "Balance Metrics"
        BAL1[Balance Accuracy<br/>Target: 100%<br/>GUARANTEED]
        BAL2[Per-Side Sum = 1.0<br/>ALGEBRAIC]
    end

    subgraph "Generalization Metrics"
        GEN1[New Accounts<br/>Target: 75-85%]
        GEN2[Few-Shot 5-10 examples<br/>Target: 70-80%]
        GEN3[Cross-Account Transfer<br/>Via shared embedding]
    end

    subgraph "Speed Metrics"
        SPEED1[Single Hypothesis<br/>Target: <100ms]
        SPEED2[Top-5 Hypotheses<br/>Target: <500ms]
        SPEED3[Scales O K linearly]
    end

    style BAL1 fill:#00ff00,stroke:#00aa00,stroke-width:3px
    style BAL2 fill:#00ff00,stroke:#00aa00,stroke-width:3px
```

## Implementation Roadmap Timeline

```mermaid
gantt
    title 8-Week Implementation Timeline
    dateFormat  YYYY-MM-DD
    section Infrastructure
    Lorentz Riemannian Optimizer           :a1, 2025-01-01, 7d
    Historical Entry Index FAISS           :a2, 2025-01-01, 7d
    Retrieval Pipeline                     :a3, 2025-01-08, 7d
    section Basic Model
    Single-Hypothesis Decoder              :b1, 2025-01-15, 7d
    Hierarchical Account Prediction        :b2, 2025-01-22, 7d
    Algebraic Balance Derivation           :b3, 2025-01-29, 7d
    Baseline Training & Evaluation         :b4, 2025-02-05, 7d
    section Multi-Hypothesis
    K Parallel Decoder Heads               :c1, 2025-02-12, 7d
    Winner-Takes-All Loss                  :c2, 2025-02-19, 7d
    Diversity Regularization               :c3, 2025-02-19, 7d
    Tuning & Evaluation                    :c4, 2025-02-26, 7d
    section Refinement
    Confidence Calibration                 :d1, 2025-03-05, 7d
    Inference Speed Optimization           :d2, 2025-03-12, 7d
    Evaluation Metrics & Monitoring        :d3, 2025-03-12, 7d
    Pilot Deployment                       :d4, 2025-03-19, 7d
```

---

## Notes on Diagram Interpretation

### Color Coding
- **Red/Pink**: Pre-trained, frozen components (embeddings)
- **Green**: Retrieval and indexing components
- **Blue**: Core neural network components
- **Yellow**: Output/predictions
- **Purple**: Loss computation
- **Cyan**: Inference pipeline

### Key Architectural Decisions Visualized
1. **Two-Stage Training**: Clear separation between embedding pre-training (Stage 1) and task learning (Stage 2)
2. **Frozen Components**: TextEncoder and LorentzEmbeddings are never updated during task training
3. **Winner-Takes-All**: Only the best hypothesis receives gradient updates
4. **Balance Guarantee**: Last line proportion is derived algebraically, not predicted
5. **Hierarchical Fallback**: Confidence thresholds determine whether to return exact, parent, or ancestor account

### Critical Paths
- **Training**: Data in Encode in Retrieve in Attend in Decode in Loss in Update
- **Inference**: Text in Encode in Retrieve in Attend in Generate K in Rank in Output
- **New Accounts**: Metadata in Parent Embedding in Exponential Map in Verify Cone

### Complexity Indicators
- **O(N)**: Account distance computation (batched)
- **O(log N)**: Tree traversal in top-down mode
- **O(K)**: Hypothesis generation (parallel)
- **O(K in )**: Diversity loss (pairwise)
