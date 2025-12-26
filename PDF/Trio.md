```markdown
# Toward Closed-loop Molecular Discovery via Language Model, Property Alignment and Strategic Search

Junkai Ji$^{1, \dagger}$, Zhangfan Yang$^{1,2, \dagger}$, Dong Xu$^{1}$, Ruibin Bai$^{2}$, Jianqiang Li$^{1}$, Tingjun Hou$^{3,*}$, and Zexuan Zhu$^{1,*}$

$^{1}$School of Artificial Intelligence, Shenzhen University Shenzhen, China
$^{2}$School of Computer Science, University of Nottingham Ningbo, Ningbo, China
$^{3}$College of Pharmaceutical Sciences, Zhejiang University, Hangzhou,China
$^{\dagger}$Equal contribution
$^{*}$Corresponding author: tingjunhou@zju.edu.cn; zhuzx@szu.edu.cn

## Abstract

Drug discovery is a time-consuming and expensive process, with traditional high-throughput and docking-based virtual screening hampered by low success rates and limited scalability. Recent advances in generative modelling, including autoregressive, diffusion, and flow-based approaches, have enabled *de novo* ligand design beyond the limits of enumerative screening. Yet these models often suffer from inadequate generalization, limited interpretability, and an overemphasis on binding affinity at the expense of key pharmacological properties, thereby restricting their translational utility. Here we present Trio, a molecular generation framework integrating fragment-based molecular language modeling, reinforcement learning, and Monte Carlo tree search, for effective and interpretable closed-loop targeted molecular design. Through the three key components, Trio enables context-aware fragment assembly, enforces physicochemical and synthetic feasibility, and guides a balanced search between the exploration of novel chemotypes and the exploitation of promising intermediates within protein binding pockets. Experimental results show that Trio reliably achieves chemically valid and pharmacologically enhanced ligands, outperforming state-of-the-art approaches with improved binding affinity (+7.85%), drug-likeness (+11.10%) and synthetic accessibility (+12.05%), while expanding molecular diversity more than fourfold.

## Introduction

Drug discovery remains an exceedingly complex, costly, and time-intensive enterprise, typically requiring over a decade of sustained effort and substantial financial investment to translate a single therapeutic candidate into a clinically approved drug. Traditional high-throughput screening approaches have made important contributions, yet they are often constrained by low hit rates, escalating experimental costs, and limited coverage of the vast chemical space [1]. Docking-based virtual screening has provided a

1
```

---

promising computational alternative, enabling the rapid prioritization of lead compounds and the identification of novel therapeutic opportunities. Nevertheless, these approaches remain hindered by high false-positive rates and intrinsic scalability bottlenecks, particularly as chemical libraries expand exponentially in both size and structural complexity [2, 3]. Recent advances in generative modeling, however, represent a paradigm shift, offering a transformative capability to design novel lead compounds under task-specific optimization constraints. By directly generating molecular structures with desired properties, this emerging strategy not only mitigates the limitations of ultra-large library screening but also enables systematic exploration of previously inaccessible regions of chemical space [4, 5].

Recent advances have introduced autoregressive generative models for designing ligands directly from protein 3D structural contexts. Representative approaches include Pocket2Mol, which leverages an E(3)-equivariant graph neural network to encode pocket geometry [6], ResGen, which integrates pocket information with fragment-based autoregression [7], and FragGen, which adopts fragment-wise generation guided by interaction graphs [8]. While these models can condition molecular generation on protein features, their strictly sequential nature deviates from physical reality, accumulating errors that frequently yield chemically implausible structures [9]. To overcome these issues, diffusion and flow-based models have emerged, offering simultaneous generation of all atoms and thereby capturing global interactions during the generative process [10]. Notable examples include DiffBP, which employs diffusion denoising consistent with physical laws [11], DiffSBDD, an SE(3)-equivariant conditional diffusion model [12], and EquiFM, an equivariant flow-matching framework designed for greater efficiency [13]. Collectively, these target-aware conditional generative models have shown promise in producing high-affinity ligands. Nevertheless, the limited availability of experimentally resolved protein–ligand complexes continues to impede model training, restricting their generalization and robustness in practical drug discovery applications [14].

To overcome the generalization limitations of protein-conditioned generative models, researchers have increasingly drawn inspiration from language models, following the success of GPT in diverse domains. Molecular structures can be expressed in textual formats such as SMILES [15], SELFIES [16], and SAFE [17], enabling ultra-large compound libraries—once a bottleneck for virtual screening—to serve as valuable training corpora for molecular language models (MLMs). Representative efforts include BindGPT, which generates 3D ligands in protein binding sites through large-scale pretraining and reinforcement learning [18]; 3DSMILES-GPT, which augments SMILES with coordinate tokens for structure-aware generation [19]; and TamGen, which incorporates protein embeddings via cross-attention and demonstrates efficacy in generating inhibitors against tuberculosis ClpP protease [20]. Despite improving generalization, current MLMs remain insufficient for precise protein-pocket targeting, and auxiliary optimization procedures often overemphasize binding affinity at the expense of drug-likeness (QED) and synthetic accessibility (SA), thereby limiting their translational utility in drug discovery [21].

In summary, while recent molecular generation models provide powerful means to navigate chemical space and design novel compounds, they often reduce molecular design to over-atomized or over-symbolized representations (Fig. 1a). By prioritizing local interactions with binding-site residues through physics-based optimization, these approaches frequently neglect the semantic integrity of molecular func-

2

---

```markdown
tionality, undermining the plausibility of whole-molecule affinity [22]. Moreover, the limited interpretability of current models remains a fundamental barrier: their black-box nature obscures the pathways of molecular optimization, leaving chemists unable to rationalize or trust the design outcomes, thereby constraining their broader adoption in drug discovery [23].

To jointly address generalization, plausibility, and interpretability, we propose Trio, a closed-loop paradigm that integrates a fragment-based MLM, reinforcement learning (RL), and Monte Carlo tree search (MCTS). At its core, a fragment-based MLM is trained on millions of SMILES strings to capture broad fragment sequence distributions and generate context-aware assemblies while circumventing the syntactic complexity of numeric junction identifiers and ring-index markers in SAFESGPT. To ensure drug-like plausibility, RL aligns the generative process with critical molecular properties such as QED and SA scores. Finally, the RL-aligned MLM then acts as the policy within MCTS, which explores fragment assembly trajectories in protein pockets using an upper confidence bound strategy to balance exploitation of promising structures against exploration of novel chemotypes, guided by affinity, pharmacokinetic, and SAR rewards. By combining fragment-level semantics, property-constrained optimization, and tree-based search, Trio achieves an interpretable and efficient molecular generation process that overcomes key limitations of prior approaches. Building upon this design, the backbone fragment-based MLM first demonstrates strong validity and novelty across both general *de novo* and constrained generation tasks. For the target-based molecular generation setting, Trio establishes a new performance benchmark, significantly outperforming state-of-the-art approaches through a superior balance of physicochemical properties. The model achieves robust gains over current baselines, improving drug-likeness by 11.10% and synthetic accessibility by 12.05%. Crucially, these enhancements are not achieved at the expense of potency; rather, Trio concurrently elevates predicted binding affinity (+7.85%) while notably expanding molecular diversity by fourfold. These results underscore the complementary strengths of fragment-informed MLMs, RL-driven property alignment, and MCTS-based strategic exploration, offering an effective and interpretable paradigm for targeted molecular design.

## Results

### Overview of Trio framework

Trio can produce desired molecules from scratch for each target protein. Its overall generation procedures can be divided into three parts: first, using self-supervised learning to train an MLM for next fragment prediction tasks; second, adopting reinforcement learning to fine-tune the MLM for customized molecular property alignment; third, leveraging the Monte Carlo Tree Search and the aligned MLM to stepwise generate molecules in three-dimensional protein pockets.

The supervised MLM of Trio uses a GPT-like architecture, named FRAGPT, to predict molecule fragments in an autoregressive manner. Original SMILES strings of molecules need to be modified into fragment-based SMILES tokens for training. The fragmentation approach not only preserves intrinsic intra-fragment semantic information but also explicitly captures the chemical interactions between frag-

3
```

---

```
## (a) Previous methods for molecule generation and evaluation

### Sequence-based methods
*   lack of 3D structure
*   semantic inconsistency
    *   O=C(C[C@H](Oc1ccccc1)c1ccccc1)N1CCCC1

### Search-based methods
*   **Total Chemical space** (depicted by a large circle of molecules)
*   **Accessible search space** (depicted by a smaller circle of molecules)
*   predefined rules (icon: key)
*   existing library (icon: key)
*   limited space, inefficient search

### Graph-based methods
*   distortion in generation
*   scarcity of paired structural data
*   PDBbind+ (associated with a protein-ligand complex image)

## (b) Our method: Machine learning Trio of MLM, DPO and MCTS

### Stage 1: MLM Broad chemical-space coverage; generates chemically valid molecules.
*   Fragment-by-fragment is ordered
*   Independent fragments—clean sequential flow

**Process:**
1.  **FragSeqs** (`c1ccc(C#N)cc1[SEP]c1occc1[SEP]C(=O)[C@H](c1ccccc1)c1ccccc1[SEP]N1CCCC1`)
    *   `-> Building`
2.  **10 M FragSeqs DATABASE**
    *   `-> Training`
3.  **FRAGPT** (Fragment Generative Pre-trained Transformer)
    *   `->` (Generated Molecular Image)
    *   `-> Step-by-Step Molecular Generation`

### Stage 2: DPO Positive/negative preference optimization; yields synthesizable, drug-like candidates.

**Process:**
1.  (Molecule Image - Better QED, Better SA) **V.s** (Molecule Image - Worse QED, Worse SA)
    *   `-> Building`
2.  **100 K DPO FragSeqs**
    *   `-> DPO`
3.  **FRAGPT-DPO**
    *   `-> Enhancing Drug-likeness in Generated Molecules`

### Stage 3: MCTS Library-free, 3D pocket-conditioned generation at the binding site.

**Legend:**
*   : Tree Node (Allow expansion)
*   : Leaf Node (Can not expansion)
*   : Selection
*   : Expansion
*   : Simulation
*   : Backpropagation

**Tree Structure (Simplified representation of the diagram):**
*   **Tree Depth 0 (Root)**
*   **Tree Depth 1** (Tree width shown vertically)
    *   Node (Mol Image): Called: 5, Reward: 6
    *   Node (Mol Image): Called: 8, Reward: 9 (Selected for expansion)
    *   Node (Mol Image): Called: 0, Reward: 3
    *   Node (Mol Image): Called: 1, Reward: 4
*   **Tree Depth 2** (Children of selected node at Depth 1)
    *   Node (Mol Image): Called: 1, Reward: 10
    *   Node (Mol Image): Called: 4, Reward: 7
    *   Node (Mol Image): Called: 9, Reward: 11 (Selected for expansion)
    *   Node (Mol Image): Called: 0, Reward: 5
*   **Tree Depth 3** (Children of selected node at Depth 2)
    *   Node (Mol Image): Called: 1, Reward: 12
    *   Node (Mol Image): Called: 4, Reward: 11
    *   Node (Mol Image): Called: 7, Reward: 13 (Selected for expansion)
*   **Tree Depth 4** (Child of selected node at Depth 3)
    *   Node (Mol Image): Called: 2, Reward: 14 (Selected, followed by Backpropagation)
*   **Tree Depth 5** (Result of Top-Ranked Route Record)
    *   Node (Mol Image): Reward: 15

**MCTS Steps:**
*   **Selection** (Orange box)
*   **Expansion** (Purple box)
*   **Simulation** (Yellow box)
*   **Backpropagation** (Blue box)

**Termination conditions:**
1.  The remaining number of extensions is 0.
2.  All expandable nodes are leaf nodes.

---
Figure 1: Overview and motivation of the proposed Trio framework. **a**, Limits of prior paradigms. Sequence-based (SMILES) models miss 3D context and inter-fragment semantics; search-based GA/MCTS depend on fixed fragment libraries and hand-crafted link rules, creating complicated and slow searches; structure-based 2D/3D generators need scarce protein-ligand pairs and risk geometric distortion. **b**, Trio pipeline. Stage 1: Pre-train: FRAGPT, a fragment language model trained on FragSeqs, learns context-aware attachments to assemble valid molecules step-by-step. Stage 2: Preference alignment, DPO with QED/SA pairs biases the policy toward synthesizable, drug-like compounds. Stage 3: Pocket-conditioned planning, the DPO-aligned policy drives MCTS with UCB over Selection-Expansion-Simulation-Backpropagation, combining affinity rewards to rank routes.

4
```

---

Here is the extracted text from the image, preserving its structure:

ments. Subsequently, our supervised MLM employs a causal attention mechanism to generate molecular fragments step by step based on their contextual semantic environment, as shown in Fig. 1b. Such a fragment-based generation strategy can effectively leverage the strong semantic feature extraction capability of LLMs and significantly reduce complexity compared to generating entire molecules directly.
However, the supervised MLM architecture alone is insufficient for generating molecules with desirable target properties, since GPT-like models inherently align with learned semantic distributions of tokens rather than explicitly optimized attributes. To address this limitation, the direct preference optimization (DPO) is used to fine-tune the supervised MLM by explicitly aligning the model's conditional distribution with an external preference signal that reflects the desired molecular attributes [24]. The strategy can incorporate property preferences into the supervised MLM, similar to the human preference alignment training for LLMs [25]. Such an aligned MLM enables the production of druggable molecules simultaneously satisfying multiple targeted properties.

Furthermore, Trio combines the aligned MLM with an MCTS algorithm in the complicated target-aware molecular design. The hybrid approach leverages MCTS's strengths in balancing exploration and exploitation, facilitating a more diverse generation of molecules with enhanced binding affinities. The flexibility is an additional advantage of this paradigm, which allows straightforward adjustment of the search objectives through altering reward functions, circumventing the computational overhead associated with repeated fine-tuning. Moreover, this fragment-by-fragment search significantly enhances interpretability compared to fine-tuning approaches, because the optimization trajectory of molecular fragments transparently reflects the strategic decision-making process whereas fine-tuning interpretability remains constrained by black-box neural network weights.

## Performance Evaluation of FRAGPT for *De novo* and Fragment-Constrained Generation Tasks

To evaluate the performance of our proposed model, FRAGPT, on molecular generation tasks, we conducted a comparative analysis against two existing fragment-based models: SAFEGPT [26] and GenMol [27]. As illustrated in Fig. 2a, a primary distinction lies in the encoding strategy: FRAGPT utilizes the FragSeq representation for molecular fragmentation, whereas both SAFEGPT and GenMol employ SAFE strings. Furthermore, the baseline models differ in their underlying architectures. Specifically, SAFEGPT is built upon the GPT framework, while GenMol leverages a diffusion language model (Fig. 2b). The evaluation framework focuses on two key aspects of generative chemistry: the capacity for creating novel chemical entities (*de novo* generation) and the precision of fragment-based structural assembly (fragment-constrained generation), with the specifics of each task outlined in Fig. 2c. The quality of the generated molecules is assessed through three complementary metrics: Validity, which quantifies the proportion of syntactically correct SMILES; Uniqueness, which measures the percentage of non-redundant molecules among the valid ones; and Diversity, calculated as the average pairwise Tanimoto distance between the Morgan fingerprints of the generated molecules. For fragment-constrained tasks, we introduce an additional metric, Distance, defined as the average Tanimoto distance between the generated molecules and the target fragments.

5

---

```markdown
# Figure 2: FRAGPT for De Novo and Fragment-Constrained Molecular Generation: Representations, Models, Tasks, and Performance.

(a) Molecule Representation

**SAFE**
*   Chemical structure illustration (benzene rings, nitrogen, oxygen, etc.)
*   Representation: `N#Cc1ccc9cc1.c1d9ccc7cc1.c1d8cccc1`
*   Indexed atoms—fragile coupling
*   Independent fragments—clean sequential flow

**FragSeq**
*   Chemical structure illustration (benzene rings, nitrogen, oxygen, etc.)
*   Representation: `c1ccc(C#N)cc1[SEP*]c1ccc(*)cc1[SEP*]C(=O)[C@H](*)c1ccccc1[SEP]N1CCCC1`
*   Position digits—error-prone encoding
*   Fragment-by-fragment—ordered and robust

(b) Generative Model

**Diffusion**
*   Random Sampling
*   Step 1: C C O
*   Step 2: C C = C O
*   Step 3: C C = C O N C

**GPT**
*   Step by Step
*   Step 1: C C
*   Step 2: C C =
*   Step 3: C C = C ...
    *   **Legend:**
        *   Blue box: Predicted token
        *   Grey box: Mask token
        *   Orange box: Given token

(c) Task taxonomy

*   **Central:** Generated Molecule
*   **Generation tasks (clockwise from top-left):**
    *   Linker generation
    *   Scaffold morphing
    *   De novo generation (leads to ∅, implying generation from scratch)
    *   Superstructure generation
    *   Motif extension
    *   Scaffold decoration

(d) De novo generation: four models compared on the core metrics

*   **Central Metrics Legend:**
    *   Validity (red bar)
    *   Uniqueness (grey bar)
    *   Diversity (blue bar)

*   **GenMol$_{1-step}$**
    *   Validity: 97.6%
    *   Uniqueness: 100%
    *   Diversity: 0.843

*   **SAFE-GPT**
    *   Validity: 94.0%
    *   Uniqueness: 100.0%
    *   Diversity: 0.879

*   **FRAGPT**
    *   Validity: 98.3%
    *   Uniqueness: 100%
    *   Diversity: 0.892

(e) Task-wise performance of three models across LD, SM, ME, SD, and SG on Validity, Uniqueness, Diversity, and Distance

*   **Model Legend:**
    *   SAFEGPT (green bar)
    *   GenMol (blue bar)
    *   FRAGPT (orange bar)

*   **Task Legend:**
    *   LD = Linker Design
    *   SM = Scaffold Morphing
    *   ME = Motif Extension
    *   SD = Scaffold Decoration
    *   SG = Superstructure Generation

*   **Charts:**
    *   **Validity:** Bar chart showing validity percentages for SG, SD, ME, SM, LD tasks across SAFEGPT, GenMol, and FRAGPT.
    *   **Uniqueness:** Bar chart showing uniqueness percentages for SG, SD, ME, SM, LD tasks across SAFEGPT, GenMol, and FRAGPT.
    *   **Diversity:** Bar chart showing diversity scores for SG, SD, ME, SM, LD tasks across SAFEGPT, GenMol, and FRAGPT.
    *   **Distance:** Bar chart showing distance scores for SG, SD, ME, SM, LD tasks across SAFEGPT, GenMol, and FRAGPT.

---

## Figure Caption:

Figure 2: **FRAGPT** for *De Novo* and Fragment-Constrained Molecular Generation: Representations, Models, Tasks, and Performance. **a**, Two fragment-based SMILES representations: SAFE and FragSeq, illustrating tokenization and ordering; **b**, Two language-model families for molecule generation: diffusion with random sampling and GPT with step-by-step masked prediction; **c**, Task taxonomy. Linker generation and scaffold morphing share the same conditional form but use different given fragments. Motif extension, scaffold decoration, and superstructure generation also share a common form, conditioned respectively on a motif, a scaffold, or a superstructure; **d**, De novo generation: four models compared on the core metrics; **e**, Task-wise performance of three models across LD (Linker Design), SM (Scaffold Morphing), ME (Motif Extension), SD (Scaffold Decoration), and SG (Superstructure Generation) on Validity, Uniqueness, Diversity, and Distance. Validity is the percentage of chemically valid molecules. Uniqueness is the proportion of unique molecules among the valid ones. Diversity measures internal structural dissimilarity within the generated set. Distance measures structural similarity to a reference molecule; values approaching 1 indicate greater dissimilarity.

---

Page number: 6
```

---

```markdown
each generated structure and its corresponding reference molecule. This metric captures the degree of chemical exploration under fixed structural constraints.

As shown in Fig. 2d, FRAGPT trained on merely 1% of the SAFE dataset achieves or even surpasses the performance of baseline models trained on the full corpus in the *de novo* generation task, demonstrating its remarkable data efficiency. The comparatively low validity of SAFEGPT arises from its reliance on positional numeric markers for fragment linkage. It scales poorly because these digits interfere with canonical ring-closure notation and elevate syntactic ambiguity with increasing fragment counts. FRAGPT avoids this failure mode by imposing a structured fragment syntax that disentangles junction semantics from ring indices, yielding both higher validity and greater structural diversity. Although the diffusion-based GenMol attains substantial validity, its conservative denoising schedule suppresses exploration, leading to a diversity deficit.

While FRAGPT demonstrates exceptional capability in *de novo* molecular generation, its true strength lies in addressing the critical need for fragment-constrained molecular design—a cornerstone of lead optimization in drug discovery. We rigorously evaluate FRAGPT across five key tasks: scaffold decoration, scaffold morphing, linker generation, motif extension, and superstructure generation. Notably, as a left-to-right autoregressive model, FRAGPT inherently faces architectural constraints in tasks requiring simultaneous satisfaction of both start- and end-fragment conditions (linker design and scaffold morphing). To overcome this challenge, we implement a novel beam search strategy that systematically explores the chemical space while maintaining fragment constraints, achieving remarkable success in these demanding scenarios.

Fig. 2e presents the results of fragment-constrained generation from 100 generated samples per task. Compared to the other methods, FRAGPT demonstrates consistent and superior performance across multiple critical metrics. Specifically, FRAGPT attains near-perfect validity in every task and delivers the highest structural-distance scores across the board, indicating a consistently broader exploration of chemical space than SAFEGPT or GenMol. Even within the structurally confined tasks of linker design and scaffold morphing, FRAGPT demonstrates remarkable generative diversity. Its resulting intermolecule distance significantly surpasses all competing methods, indicating that its generated candidates populate more distant and novel regions of the chemical space. In less constrained generation settings such as motif extension, scaffold decoration, and superstructure elaboration, the model consistently achieves high uniqueness, broad exploration radius, and strong chemical fidelity, demonstrating both flexibility and precision.

## Verification of DPO algorithm for Drug-likeness Property Alignment

After the successful *de novo* and fragment-constrained generation experiments, we employ the DPO algorithm to align our FRAGPT model with the drug-like scores in preparation for subsequent target-specific molecule-generation tasks. Here, the drug-likeness is quantified by the QED and SA metrics. To characterize the explored chemical space, we sampled 10,000 molecules from the training set and independently generated 10,000 molecules with SAFEGPT, vanilla FRAGPT, and DPO-aligned FRAGPT. In this experiment, MACCS fingerprints were embedded into a two-dimensional manifold via t-distributed

7
```

---

Figure 3: Comparative characterization of generated chemical spaces across baseline data and generative models.
a, Two-dimensional t-SNE of MACCS fingerprints of 10000 generated molecules per set, showing pairwise overlaps between DATASET, FRAGPT, SAFEGPT and FRAGPT-DPO.
Titles:
*   DATASET vs FRAGPT
*   DATASET vs FRAGPT-DPO
*   DATASET vs SAFEGPT
*   FRAGPT vs FRAGPT-DPO
*   FRAGPT vs SAFEGPT
*   FRAGPT-DPO vs SAFEGPT
Legend: DATASET, FRAGPT, FRAGPT-DPO, SAFEGPT

b, Box plots of drug-likeness (QED) and synthetic accessibility (SA) for the same sets.
*   QED Distribution Across Models
    *   Y-axis: 0.00, 0.25, 0.50, 0.75, 1.0
    *   X-axis labels: DATASET, SAFEGPT, FRAGPT, FRAGPT-DPO
*   SA Distribution Across Models
    *   Y-axis: 0.00, 0.25, 0.50, 0.75, 1.00
    *   X-axis labels: DATASET, SAFEGPT, FRAGPT, FRAGPT-DPO

c, Hexbin density maps of the QED-SA landscape.
*   DATASET
    *   Y-axis (SA): 0.0, 0.5, 1.0
    *   X-axis (QED): 0.0, 0.5
    *   Right Y-axis (Frequency): 0.000, 0.001, 0.002, 0.003
*   SAFEGPT
    *   Y-axis (SA): 0.0, 0.5, 1.0
    *   X-axis (QED): 0.0, 0.5
    *   Right Y-axis (Frequency): 0.000, 0.001, 0.002, 0.003
*   FRAGPT
    *   Y-axis (SA): 0.0, 0.5, 1.0
    *   X-axis (QED): 0.0, 0.5
    *   Right Y-axis (Frequency): 0.000, 0.001, 0.002, 0.003
*   FRAGPT-DPO
    *   Y-axis (SA): 0.0, 0.5, 1.0
    *   X-axis (QED): 0.0, 0.5
    *   Right Y-axis (Frequency): 0.000, 0.001, 0.002, 0.003

d, Statistical analysis of generated molecular substructures. A comparison of atom, bond, and ring distributions between the reference dataset and molecules from three generative models.
Legend: DATASET, SAFEGPT, FRAGPT, FRAGPT-DPO

*   (Top) Relative frequency plots show the proportion of each substructure category within each data source.
    *   Y-axis: 0.00, 0.25, 0.50, 0.75
    *   Atom Type: C, N, O, P, F, S, Cl, Br, other
    *   Bond Type: C-C, C=C, C-N, C=N, C-O, C=O, C:C, C:N, other
    *   Ring Size: 3, 4, 5, 6, 7, 8, 9, other

*   (Bottom) Normalized count plots compare the prevalence of each substructure across the different sources, with values for each category scaled by the maximum observed count.
    *   Y-axis: 0.0, 0.5, 1.0
    *   Atom Type: C, N, O, P, F, S, Cl, Br, other
    *   Bond Type: C-C, C=C, C-N, C=N, C-O, C=O, C:C, C:N, other
    *   Ring Size: 3, 4, 5, 6, 7, 8, 9, other

Page number: 8

---

stochastic neighbour embedding (t-SNE).

As illustrated in Fig. 3a, vanilla FRAGPT almost completely spans the data manifold of the training set. The DPO-aligned variant (termed FRAGPT-DPO) further concentrates this distribution toward the data-dense core, whereas SAFEGPT preserves the central cloud but generates several additional high-density clusters that are sparsely represented in the original dataset, likely owing to its larger and more diverse training corpus. FRAGPT-DPO tends to compress the existing distribution and shift sample density inward, while SAFEGPT produces several new high-density clusters absent from the FRAGPT-DPO landscape.

To verify whether the contraction of FRAGPT-DPO’s output distribution would compromise fundamental chemical realism, we compared the drug-like property distributions (QED and SA) across the aforementioned statistics. For the distribution of QED and SA, Figs. 3b and 3c reveal that vanilla FRAGPT closely mirrors the joint QED-SA landscape of the training set. Meanwhile, SAFE attains an improvement in the QED compared with vanilla FRAGPT, yet its SA distribution remains broader, suggesting a bias toward drug-likeness over synthetic accessibility. After DPO alignment, FRAGPT-DPO shows a clear upward shift in QED and a moderate increase in SA, accompanied by a contraction in SA variance. The hex-bin plot reveals a marked shift in sample density toward the chemically desirable region, effectively eliminating the low-quality long tail present in the original data.

Besides basic drug properties, we also compared the relative frequencies of fundamental structural descriptors generated by three language-based molecular models with those in the training data. As shown in Fig. 3d, the top panel demonstrates that all three generators closely reproduce the training-set statistics for atom-type, bond-type, and ring-size distributions. The bottom panel further reveals that vanilla FRAGPT preserves similar frequencies of all three descriptors, including low-frequency halogens (I, Br, Cl) and macrocycles. While this feature expands structural diversity, it also causes a decline in the SA and QED scores of the generated molecules. In contrast, SAFE retains too many small and large ring structures, resulting in inferior QED and SA compared to FRAGPT-DPO. Notably, FRAGPT-DPO abandons chemically unfavourable motifs, thereby improving drug-likeness and synthetic accessibility relative to the dataset.

## Performance Assessment of Trio for Target-Specific Molecule Design Problem

A pivotal task in molecular design is protein-targeted molecule generation, aiming to generate entirely novel compounds exhibiting improved binding affinity toward specific protein targets. Inspired by recent experimental frameworks outlined in [28], we conducted a comprehensive evaluation of Trio across a series of protein-targeted molecular generation tasks. These tasks involve optimizing the molecular docking score (DS) for five well-established protein targets: parp1, fa7, 5ht1b, braf, and jak2. Simultaneously, we assessed whether the generated molecules maintain desirable drug-like properties, quantified using QED, SA, and chemical novelty.

9

---

Table: Quantitative comparison of docking performance on five protein targets.

| Group | Method | parp1 ↓ Mean | parp1 ↓ Std | fa7 ↓ Mean | fa7 ↓ Std | 5ht1b ↓ Mean | 5ht1b ↓ Std | braf ↓ Mean | braf ↓ Std | jak2 ↓ Mean | jak2 ↓ Std |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baselines | JT-VAE [29] | -9.482 | 0.132 | -7.683 | 0.048 | -9.382 | 0.332 | -9.079 | 0.069 | -8.885 | 0.026 |
| | REINVENT [30] | -8.702 | 0.523 | -7.205 | 0.264 | -8.770 | 0.316 | -8.392 | 0.400 | -8.165 | 0.277 |
| | Graph GA [31] | -10.949 | 0.532 | -7.365 | 0.326 | -10.422 | 0.670 | -10.789 | 0.341 | -10.167 | 0.576 |
| | MORLD [32] | -7.532 | 0.260 | -6.263 | 0.165 | -7.869 | 0.650 | -8.040 | 0.337 | -7.816 | 0.133 |
| | HierVAE [33] | -9.487 | 0.278 | -6.812 | 0.274 | -8.081 | 0.252 | -8.978 | 0.525 | -8.285 | 0.370 |
| | GA+D [34] | -8.365 | 0.201 | -6.539 | 0.297 | -8.567 | 0.177 | -9.371 | 0.728 | -8.610 | 0.104 |
| | MARS [35] | -9.716 | 0.082 | -7.839 | 0.018 | -9.804 | 0.073 | -9.569 | 0.078 | -9.150 | 0.114 |
| | GEGL [36] | -9.329 | 0.170 | -7.470 | 0.013 | -9.086 | 0.067 | -9.073 | 0.047 | -8.601 | 0.038 |
| | RationaleRL [37] | -10.663 | 0.086 | -8.129 | 0.048 | -9.005 | 0.155 | No hit found | | -9.398 | 0.076 |
| | FREED [38] | -10.579 | 0.104 | -8.378 | 0.044 | -10.714 | 0.183 | -10.561 | 0.080 | -9.735 | 0.022 |
| | PS-VAE [39] | -9.978 | 0.091 | -8.028 | 0.050 | -9.887 | 0.115 | -9.637 | 0.049 | -9.464 | 0.129 |
| | MOOD [40] | -10.865 | 0.113 | -8.160 | 0.071 | -11.145 | 0.042 | -11.063 | 0.034 | -10.147 | 0.060 |
| | RetMol [41] | -8.590 | 0.475 | -5.448 | 0.688 | -6.980 | 0.740 | -8.811 | 0.574 | -7.133 | 0.242 |
| | Genetic GFN [42] | -9.227 | 0.644 | -7.288 | 0.433 | -8.973 | 0.804 | -8.719 | 0.190 | -8.539 | 0.592 |
| Recent SOTA | GEAM [43] | -12.891 | 0.158 | -9.890 | 0.116 | -12.374 | 0.036 | -12.342 | 0.095 | -11.816 | 0.067 |
| | f-RAG [28] | _-12.945_ | _0.053_ | _-9.899_ | _0.205_ | _-12.670_ | _0.144_ | _-12.390_ | _0.046_ | _-11.842_ | _0.316_ |
| Ours | **Trio*** | **-13.129** | **0.049** | **-10.359** | **0.029** | **-12.954** | **0.020** | **-12.591** | **0.034** | **-11.855** | **0.027** |
| | Trio | -12.730 | 0.012 | -10.132 | 0.015 | -12.669 | 0.039 | -12.389 | 0.009 | -11.806 | 0.018 |

Performance is assessed based on the mean docking score (DS) of generated molecules. Lower scores indicate better predicted binding affinity. In each column, **Bold** denotes the best-performing method and *italics* denote the second-best.

---

To rigorously benchmark Trio’s performance, we compared it against multiple state-of-the-art base-line methods representing various molecular generative strategies. Specifically, the comparative study encompassed four methodological families. Fragment-based approaches, including JT-VAE [29], HierVAE [33], MARS [35], RationaleRL [37], FREED [38], PSVAE [39], f-RAG [28] and GEAM [43], construct an explicit fragment vocabulary from molecular data and subsequently assemble these fragments into novel candidates. Genetic-algorithm variants, such as Graph GA [31], GEGL [36] and Genetic GFN [42], exploit fragment information through fragment-based crossover operations, whereas GA+D [34] applies a discriminator-augmented GA directly to SELFIES representations. RL baselines contain REINVENT [30] which operates on SMILES strings, and MORLD [32] which utilizes molecular graphs. Finally, MOOD [40], a diffusion-based generative model with out-of-distribution control, targets enhanced chem-ical novelty.

The primary evaluation metric employed in this experiment is the top-hit 5% score, defined as the mean DS of the top 5% unique and novel generated hits. In contrast to methods such as f-RAG [28] and GEAM [43], which are trained or fine-tuned specifically on relatively small-scale datasets like ZINC250K, FRAGPT is pretrained on a large-scale molecular corpus without relying on any curated datasets for downstream adaptation. Accordingly, we define novel hits with three strict criteria to emphasize both affinity and drug-likeness: DS < median DS of actives, QED > 0.5, and SA < 5.0.

We generated 3,000 candidate molecules for each protein target and benchmarked them against state-of-the-art baseline generators. As shown in Table 1, the foundational Trio\* model (operating without DPO constraints) achieves the best binding affinity on five targets, outperforming all baselines. This superiority demonstrates the effectiveness of coupling a generative fragment-level language model with a guided tree search procedure. While previous distribution-learning methods (e.g., [29, 33, 39, 40]) often lack target-oriented generation capability, and rule-constrained algorithms (e.g., GEAM, f-RAG) limit search efficiency, Trio\* leverages the MLM’s inherent generalization to propose diverse, chemically meaningful fragments. Crucially, MCTS utilizes the MLM’s generative probabilities to intelligently prioritize exploration paths most likely to improve docking scores, enabling Trio\* to converge efficiently toward high-affinity candidates without relying on rigid heuristics.

Building on the robust search capability of Trio\*, the full Trio framework integrates FRAGPT-DPO with MCTS to create a holistic solution for drug-like molecule search. Unlike the exploration-focused Trio\*, the full Trio model does not optimize solely for binding affinity but navigates a multi-objective landscape to prioritize drug-likeness and synthetic accessibility. Consequently, Trio maintains a com-petitive DS relative to previous state-of-the-art methods while significantly enhancing pharmacological properties. To mitigate inflated performance caused by clusters of near-identical molecules, we compared 3,000 molecules per target generated by GEAM, Trio\*, and Trio, computing the Morgan-Tanimoto simi-larity coefficient to discard any pair with a similarity greater than 0.4. Because GEAM optimizes within the limited ZINC250K database, nearly half of its molecules were removed. In contrast, even after remov-ing structurally redundant pairs, both Trio\* and Trio retained over 70% of their generated candidates, highlighting their generative breadth. As summarized in Fig. 4a, while Trio\* exhibits the most extreme DS distribution, the full Trio model achieves superior and tightly clustered values for QED and SA driven

11

---

Figure 4: Performance and Diversity Analysis on Five Therapeutic Targets. This figure evaluates the effectiveness and diversity of molecules generated by our proposed models, Trio* and Trio, against several baseline methods.

(a) Box plots comparing the distributions of Vina Docking Score (top), Quantitative Estimate of Drug-likeness (QED, middle), and Synthetic Accessibility (SA, bottom) for molecules generated by GEAM, Trio*, and Trio.
    *   **Targets (Column Headings):** 5ht1b, braf, fa7, jak2, parp1
    *   **Y-axis Metrics (Rows):**
        *   Vina Dock (with scale 5, 10, 15)
        *   QED (with scale 0.25, 0.50, 0.75)
        *   SA (with scale 0.6, 0.8)
    *   **X-axis Methods (for each plot):** GEAM, Trio*, Trio

(b) Hyperparameter sensitivity analysis for Trio* and Trio. The plots show the average Vina Docking Score from 20 independent runs as a function of varying Search Steps (top) and Search Width (bottom).
    *   **Targets (Column Headings):** 5ht1b, braf, fa7, jak2, parp1
    *   **Y-axis:** Dock Score (with scale 7.5, 10.0)
    *   **X-axis (Top Row):** Search Steps (with scale 25, 150, 350, 500)
    *   **X-axis (Bottom Row):** Search Widths (with scale 2, 8, 14, 20)
    *   **Legend:** Trio*, Trio

(c) Molecular diversity analysis using the #Circles metric. Diversity is quantified by calculating the maximum number of molecules that can be selected from a generated set of 3,000, such that every pair of selected molecules exceeds a minimum distance threshold. A higher #Circles value signifies greater diversity and exploration of the chemical space.
    *   **Targets (Column Headings):** 5ht1b, braf, fa7, jak2, parp1
    *   **X-axis:** #Circles (with scale 0, 500)
    *   **Methods (Y-axis Legend):**
        *   Trio*
        *   Trio
        *   GEAM
        *   GEAM-static
        *   REINVENT
        *   MORLD
        *   HierVAE
        *   RationaleRL
        *   FREED
        *   PS-VAE
        *   MOOD

Page 12

---

Here's the extracted text from the image, formatted for easy conversion to Markdown:

(a)

| | Leaf Node | | [BOS] | | Best Leaf Node | | root |
|---|---|---|---|---|---|---|---|
| Layer 1 | Vina: 7.1 | Vina: 8.3 | Vina: 6.1 | Vina: 8.4 | Vina: 9.3 |
| Layer 2 | Vina: 6.3 | Vina: 6.4 | Vina: 9.3 | Vina: 6.5 | Vina: 8.4 | Vina: 7.2 | Vina: 10.3 |
| Layer 3 | Vina: 8.6 | Vina: 6.1 | Vina: 11.7 | Vina: 10.2 | Vina: 6.3 | Vina: 10.2 |
| Layer 4 | | | Vina: 10.3 | Vina: 10.5 | | Vina: - kcal/mol |
| Layer 5 | Vina: 12.3 | Vina: 11.8 | Vina: 10.5 | Vina: 9.9 | |

(b)

5ht1b
bref
fa7
jak2
parp1

| Target name | Reference Vina | Trio Vina |
|:------------|:---------------|:----------|
| 5ht1b       | -8.78 kcal/mol | -14.9 kcal/mol |
| braf        | -10.3 kcal/mol | -14.6 kcal/mol |
| fa7         | -8.5 kcal/mol  | -11.5 kcal/mol |
| jak2        | -9.1 kcal/mol  | -12.6 kcal/mol |
| parp1       | -10.0 kcal/mol | -14.5 kcal/mol |

Hydrogen Bond
π-stacking
Hydrophobic Interaction

Figure 5: Illustration of the Trio framework's stepwise generative mechanism and the intermolecular interactions between generated ligands and target protein binding pockets.
**a**, Schematic illustration of the Monte Carlo Tree Search for target-based _de novo_ generation. Starting from the [BOS] root token, molecules are constructed via iterative fragment addition (Layers 1-5) and prioritized by AutoDock Vina scores to identify the optimal candidate (crown icon); **b**, Predicted binding modes of generated leads against target proteins. Detailed views of the binding pockets for 5ht1b, braf, fa7, jak2, and parp1 highlight key non-covalent interactions. Contacts are color-coded: hydrophobic (warmpink dashed), hydrogen bonds (forestgreen solid), and π-π stacking (teal dashed).

13

---

by preference alignment, offering the optimal balance for practical drug discovery.

To assess how the number of Monte-Carlo simulations and the tree width affect MCTS performance, we analyzed the resulting docking scores across varying simulation counts and tree widths. As shown in Fig. 4b, docking scores generally improve as the simulation count increases. By contrast, expanding the tree width enhances exploration but yields no statistically significant gain in docking performance.

To further assess chemical space coverage of generated molecules, we adopt the #Circles metric as introduced in [44], which identifies distinct chemical clusters by iteratively removing molecules with high similarity (Morgan-Tanimoto similarity > 0.75). As shown in Fig. 4c, conventional distribution-learning models (e.g., HierVAE and MORLD) tend to suffer from mode collapse, generating outputs clustered near the training distribution. Although hybrid approaches such as GEAM incorporate a genetic algorithm to refine candidate molecules, they remain fundamentally constrained by predefined fragment libraries. This reliance on static building blocks inherently limits the diversity of generated molecules, restricting exploration to the chemical space spanned by the initial library. In contrast, the Trio* model demonstrates a significant multi-fold improvement in #Circles across all five protein targets, reflecting its unconstrained capacity for exploration. The full Trio model exhibits an expected moderate reduction relative to Trio due to the constraints of preference alignment, yet its #Circles count remains superior to earlier methods. This significant boost highlights the ability of our method to overcome the limitations of rule-based search and static fragment libraries, enabling more diverse and novel molecular generation. Crucially, our advantage is consistent across all targets regardless of receptor type or structural complexity, suggesting that the combination of MLM and tree search robustly generalizes across different biological contexts. This consistency circumvents the target transferability issues frequently observed in purely data-driven or rule-constrained methods, demonstrating the unique adaptability of Trio in navigating diverse chemical landscapes.

Trio’s synergistic architecture is depicted as an MCTS-guided hierarchical search procedure in Fig. 5a, which seamlessly integrates the extensive semantic knowledge encoded in FRAGPT with the established search efficiency of MCTS. Specifically, pretrained on large-scale and unlabeled molecular corpora, FRAGPT can serve as a dynamic and chemically expressive source of fragment proposals, in sharp contrast to methods limited by static, predefined fragment libraries. By capturing the syntactic and semantic regularities of chemical structures, it generates fragments that are simultaneously diverse, synthetically plausible, and chemically coherent. MCTS then navigates this expansive LM-generated chemical space, striking an effective balance between exploitation (intensively refining trajectories that yield high docking scores) and exploration (probing less obvious regions to avoid local optima and uncover novel scaffolds). This navigation is jointly informed by the MLM’s generative likelihoods and real-time docking evaluations, thereby establishing a robust, closed-loop framework that tightly couples generative language modeling with target-aware molecular optimization.

The comprehensive visualization of the entire search tree affords a level of interpretability rarely attained in *de novo* molecular design. By explicitly charting the generative trajectory, Fig. 5a enables researchers to systematically trace the evolutionary lineage of candidate molecules, revealing the step-wise incorporation of chemical features and structural motifs that recurrently enhance predicted binding

14

---

affinity. This granular transparency transcends the mere presentation of final optimized compounds, pro- viding a mechanistic framework that elucidates how specific functional groups and fragment combinations contribute to ligand potency. Moreover, it highlights fragment-level binding propensities within diverse chemical contexts, thereby deepening our understanding of structure-activity relationships. Collectively, this interpretable design paradigm empowers medicinal chemists with actionable insights, offering a more rational, human-in-the-loop workflow that bridges generative modeling with expert-driven drug discovery.

Complementing the trajectory analysis, Fig. 5b substantiates Trio’s target-aware *de novo* design capability through structural validation across diverse protein binding pockets. The interaction analyses reveal that these compounds achieve exceptionally favorable predicted binding free energies and engage in key noncovalent interactions, such as directional hydrogen bonds, $\pi$-$\pi$ stacking, and hydrophobic contacts. Notably, a comparative table within Fig. 5b demonstrates that the Vina scores of Trio-generated ligands substantially surpass those of reference compounds across multiple target pockets, with an average increase of 46.0%. This comprehensive interaction profiling corroborates the model’s ability to generate chemically valid, synthetically accessible ligands with enhanced specificity and predicted affinity.

To sum up, the robustness of Trio can be attributed to the fact that a fragment-level language model provides chemically meaningful and highly diverse candidates, and a target-aware search loop (e.g., MCTS) guided by docking feedback selectively amplifies pathways that improve binding affinity in real time. This closed loop simultaneously broadens the accessible chemical space and accelerates conver- gence to high-quality candidates, compared to earlier pipelines that relied on static fragment libraries, hand-crafted genetic operators, or one-shot scoring. As such, the consistently superior docking energies highlight not only the effectiveness of FRAGPT itself, but also the broader promise of combining large- scale chemical language models with adaptive search as a general approach for target-specific molecular design.

# Conclusion

In this work, we introduce Trio, a generative framework that establishes a new paradigm in *de novo* drug discovery. Traditional approaches—whether atomic-level modeling or constrained fragment li- braries—have long faced inherent trade-offs between generative diversity, multi-objective optimization, and molecular plausibility. Trio overcomes these limitations by synergistically integrating a pre-trained, property-aligned fragment-based language model (FRAGPT) with Monte Carlo Tree Search (MCTS).

Through a simplified yet expressive fragment representation, FRAGPT captures the intrinsic syntax of chemical assembly, while direct policy optimization (DPO) aligns molecule generation with essen- tial pharmacological properties such as drug-likeness, synthetic accessibility, and binding affinity. When embedded within MCTS, this policy enables a guided exploration of chemical space that intelligently bal- ances exploitation of promising scaffolds with the discovery of novel chemotypes, achieving both diversity and target specificity.

Comprehensive benchmarks demonstrate that Trio consistently generates chemically valid, unique, and pharmacologically superior molecules across general and target-specific optimization tasks, surpass-

15

---

ing prior generative and docking-based methods. Insights from the Trio* further underscore the necessity of our hybrid approach: while unconstrained exploration effectively maximizes binding affinity, the full Trio framework is essential for harmonizing potency with the rigorous demands of drug-likeness and synthetic accessibility. By combining context-aware fragment modeling, property-constrained reinforcement learning, and principled combinatorial search, Trio establishes a closed-loop, interpretable, and scalable framework for molecular design. Beyond these capabilities, Trio embodies a paradigm shift in AI-driven drug discovery: it transforms how chemical space can be navigated, enabling systematic, multi-objective exploration that is both interpretable and practically actionable. Future extensions may incorporate retrosynthetic reasoning, more sophisticated ADMET-informed reward functions, and expanded fragment vocabularies, further advancing the framework’s capacity to tackle previously intractable biological targets. Ultimately, Trio sets a foundation for autonomous, closed-loop discovery systems, charting a path toward the next generation of rational, AI-guided therapeutics.

# Methods

## Datasets

The ZINC [45] and UniChem [46] databases collectively contain over 1.1 billion SMILES strings. From these, SAFE [26] carefully constructs a diverse set of molecule types into a unified database, spanning drug-like compounds, peptides, multi-fragment molecules, polymers, reagents, and non-small molecules.

To reduce the computational burden, only about 15 million SMILES strings were randomly sampled for training Trio from the SAFE dataset without applying additional data augmentation techniques.

Following the methodology of Marco et al. [47], we applied the Breaking of Retrosynthetically Interesting Chemical Substructures (BRICS) algorithm to molecules, fragmenting each from left to right into multiple FragSeqs (Fig. 2a). BRICS defines 16 chemical environments that flexibly determine suitable bond cleavage sites and retained functional groups (e.g., aromatic rings and cyclic structures), thereby generating a variety of synthetically feasible fragments [48]. To facilitate the reconstruction of complete compounds from these fragments, BRICS attaches dummy atoms at each cleavage site, marking the points at which fragments can be rejoined. By following these cleavage labels, the original molecular structure can be reconstructed from the FragSeq. In total, this dataset contains around 10 million FragSeqs.

## FRAGPT Architecture

FragSeqs consist of sequentially arranged SMILES-based fragments, from which the complete molecular SMILES strings can be reconstructed. Structurally, these FragSeqs closely resemble natural language sentences, enabling the application of advanced NLP techniques for chemical structure generation. To tokenize the FragSeqs, the model employs a widely-used tokenizer based on a regular expression pattern tailored for SMILES syntax [49]. This tokenizer is designed to capture the atomic and functional group semantics inherent in SMILES strings. The resulting vocabulary consists of approximately 600.

16

---

unique tokens, encompassing not only standard chemical tokens (e.g., atoms, bonds, branches, and ring symbols) but also all required special tokens such as [BOS] (Beginning-Of-Sequence), [EOS] (End-Of-Sequence), [SEP] (Fragment identifiers) and [PAD] (Padding indicators). Such a tokenization scheme ensures consistent encoding of molecular fragments while preserving syntactic validity.

To exploit the state-of-the-art performance of current language models, we developed a GPT-based MLM, termed FRAGPT, which is a decoder-only transformer architecture comprising 87.3 million parameters. Its architecture is specifically tailored for the generation of FragSeqs. First, the input sequence tokens are linearly projected into Query $Q$, Key $K$, and Value $V$ matrices. Subsequently, the standard self-attention mechanism computes the dot product between the Query and Key matrices, and the resulting scores are normalized via a softmax function to produce attention weights for the Value matrix. Formally, given input embeddings **X** $\in \mathbb{R}^{n \times d}$, the computational procedure can be formally represented as follows:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \quad (1)
$$

where $T$ indicates transposition. $Q = XW_Q$, $K = XW_K$ and $V = XW_V$ represent learned projections of the input embeddings $X$. A feed-forward network is then applied to the resulting output embeddings to obtain the semantic features of each token. This attention mechanism enables contextual modeling of fragment interactions through dot product operations.

During the training phase, FragSeqs are first embedded into feature matrices, after which positional embeddings (implemented via Rotary Position Embedding, RoPE) are incorporated [50]. The combined embeddings, denoted as $X$, are then fed into the FRAGPT model. The output of FRAGPT is processed through a softmax layer, where each token predicts the next token via a masking mechanism designed to prevent information leakage from future positions. The training objective is to minimize the cross-entropy loss between the model's predicted token probability distribution and the true distribution of target tokens. Formally, given a FragSeqs $Y = \{y_1, y_2, \dots, y_t\}$, the loss function can be described as:

$$
L = -\sum_{t=1}^{T} \log P(y_t|x, y_{<t}) \quad (2)
$$

where $P(y_t|x, y_{<t})$ denotes the conditional probability of the target token $y_t$, given the input embedding $x$ and all preceding tokens $y_{<t}$. FRAGPT is trained using the AdamW optimizer with hyperparameters $\beta_1 = 0.9$ and $\beta_2 = 0.95$. The training is conducted on six NVIDIA A6000 GPUs for a total of 8 epochs, employing a learning rate scheduling strategy that combines an initial warm-up phase with a subsequent linear decay. The batch size is set to 32 samples per GPU, ensuring stable optimization and efficient utilization of computational resources.

# Reinforcement Learning

Inspired by the success of GPT models enhanced through reinforcement learning (RL) algorithms for a variety of NLP tasks, we employ FRAGPT as the base model and integrate different RL approaches

17

---

tailored to each downstream task. This strategy leverages the strengths of RL-based optimization to further improve the performance and adaptability of the base model in specific applications.

**Direct Preference Optimization**: To encourage FRAGPT to generate more reasonable molecules, we adopt the DPO algorithm to smoothly align the model towards higher QED and lower SA, instead of using Augmented Likelihood Reinforcement Learning, which collapses the output distribution into peaky modes over desirable properties [51]. Unlike Proximal Policy Optimization (PPO) [52], which requires training an auxiliary reward model, DPO treats the GPT policy as the reward model. This design yields an explicit mapping between policy logits and reward signals, allowing the language model to satisfy user-defined preferences without extra critics. The general DPO pipeline employed in our experiments can be summarized as follows:

*   Generate roughly 100,000 FragSeqs using the policy initialized with FRAGPT ($\pi_{\text{ref}}$):
    $$y \sim \pi_{\text{ref}}(\cdot | x), \tag{3}$$
    where $y$ denotes a generated FragSeq, and $x$ represents its prior fragment context, i.e., the sequence of fragments generated up to that point. Specifically, we group each FragSeq by its prefix fragment and restrict every molecule to contain no more than three identical prefix fragments. When a group lacks a common prefix fragment or contains fewer than 8 molecules, we assign [BOS] as the default prefix fragment. Next, we annotate each FragSeq with its calculated pharmacological attributes (e.g., QED and SA). To construct meaningful preference pairs, we first rank the molecules in each group by their drug attributes. We then draw positive-negative pairs from the top and bottom of each ranking, thus obtaining an offline preference dataset that shares a common fragment prefix yet spans a broad range of difficulty levels:
    $$\mathcal{D} = \{(x^{(i)}, y_g^{(i)}, y_l^{(i)})\}_{i=1}^N, \tag{4}$$
    where $y_g^{(i)}$ and $y_l^{(i)}$ denote the FragSeqs derived from the same prior fragment $x^{(i)}$ that exhibit higher and lower drug-property scores, respectively. This DPO dataset construction method is specifically tailored to FragSeqs and closely mirrors the original DPO procedure used in NLP. The initial fragment sequences act as prompts, enabling the model to distinguish the relative drug-property quality of two molecules derived from the same starting fragment.
*   Maximize the likelihood of the reinforced MLM $\pi_\theta$ with respect to the reference policy $\pi_{\text{ref}}$, where the optimization objective is given by:
    $$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,y_g,y_l)\sim\mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_g|x)}{\pi_{\text{ref}}(y_g|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right], \tag{5}$$
    where $\sigma$ is the sigmoid function, and $\beta$ is a scaling coefficient that adjusts the trade-off between enhancing preference and preserving the original distribution during training.

18

---

# Monte Carlo Tree Search

Monte Carlo Tree Search (MCTS) is a heuristic search algorithm renowned for its efficacy in sequential decision-making tasks, particularly in domains requiring combinatorial optimization, most notably in AlphaGo [53]. In the context of molecule generation, MCTS operates as an iterative, tree-based framework that balances exploration of potential chemical spaces with exploitation of promising molecular candidates. The algorithm comprises four canonical phases: Selection, Expansion, Simulation and Back-propagation (Fig. 1b). It iteratively grows a decision tree to guide fragment assembly by searching for the optimal fragment sequence, identifying the most promising molecular candidates. Below, we elaborate on each phase and its corresponding role in the molecular generation pipeline:

*   **Selection Phase:** Navigating the Chemical Decision Tree. The algorithm begins with a predefined root node, which may represent a [BOS] token or an initial molecular fragment encoded in SMILES notation. The selection strategy employs a modified Upper Confidence Bound for Trees (UCT) criterion to select a child node with high potential rewards while maintaining diversity in exploration [54]. The UCT value for a child node *j* is formulated as:

    ```
    UCT_j = α × average(a_j) + (1 − α) × max(a_j) + C * sqrt(ln N_C / N_j)   (6)
    ```

    where `average(a_j)` and `max(a_j)` represent the average and maximum rewards of action `a_j`, respectively. `α` manipulates the trade-off between historical performance `average(a_j)` and optimistic potential `max(a_j)`, and `C` represents the exploration-exploitation balance by scaling the second term derived from the UCT framework. `N_C` is the total visitation count of the parent node, and `N_j` is the visitation count of node `j`. This dual-objective formulation ensures that under-explored nodes with high variance in rewards receive attention, thereby mitigating premature convergence to suboptimal regions.

    To further investigate the potential nodes, a children-adaptive strategy, as proposed by Tian et al. [55], is employed to dynamically adjust the branching factor of nodes based on their reward stability. The importance metric `I(s_t)` for the node `s_t` is calculated as:

    ```
    I(s_t) = max_{o_t^i} |R(s_t, o_t^i) - R̄(s_t)|   (7)
    ```

    where `R(s_t, o_t^i)` is the reward of the i-th child of `s_t`, and `R̄(s_t)` is the mean reward across all children. Intuitively, a high `I(s_t)` indicates significant reward deviation among children, promoting the algorithm to expand the number of child nodes to `n(s_t) = min(β|I(s_t)|, c_max)`, where `β` scales the expansion rate and `c_max` imposes an upper bound to prevent computational overload. This adaptive mechanism ensures that nodes with fluctuating reward distributions require deeper exploration, enhancing the likelihood of discovering high-reward molecular candidates.

*   **Expansion Phase:** Probabilistic Generation of Next Molecular Fragments. After selecting a leaf node, the algorithm first evaluates its terminal status. If the leaf node contains an [EOS] token,

19

---

the process returns to the selection phase. Otherwise, FRAGPT acts as the agent to generate the subsequent fragment of the SMILES sequence, conditioned on the current molecular state derived from the chemical context of the parent node. During this phase, FRAGPT generates only the next fragment SMILES by expanding a single branch from the selected node, rather than producing the entire sequence until the [EOS] token is reached. The generated fragment is appended to the current SMILES string, extending the molecular context and creating a new child node in the decision tree. More importantly, the expansion stage incorporates a duplicate detection mechanism, which calculates the molecular similarity between the current node and its sibling nodes. To avoid redundant exploration, the expansion is repeated up to five times until a structurally distinct molecule is obtained, thereby enhancing both the diversity of candidates and the overall efficiency of the optimization process.

- **Simulation Phase:** Rollout Strategies and Reward Estimation. The simulation phase evaluates the long-term potential of the newly expanded node by performing Monte Carlo rollouts until a terminal state ([EOS]) is reached. During the rollout process, the FRAGPT generates the integral candidate SMILES strings based on the current node state. In contrast to the expansion phase, the simulation phase treats FRAGPT as a simulator that generates the complete SMILES sequence and reconstructs the corresponding molecule, approximating the potential molecular state of the currently expanded node for subsequent evaluation. The resulting molecule is scored using a domain-specific reward function R(.), which quantifies desirable properties such as synthetic accessibility (SA), quantitative estimate of drug-likeness (QED), and target-specific bioactivity (e.g., docking scores). To enhance robustness, the reward function can be designed to incorporate ensemble evaluations, integrating multiple scoring functions that reflect diverse molecular objectives, facilitating more reliable assessments and supporting multi-objective optimization during molecular generation.

- **Backpropagation Phase:** Reward Dissemination and Tree Updates. The final reward R obtained from the simulation is propagated backward through the tree to update the statistics of all traversed nodes. Each node’s visitation count N_j and cumulative reward Q_j are incremented as:

N_j ← N_j + 1, Q_j ← Q_j + R. (8)

This update mechanism enables the algorithm to accumulate experience over time, reinforcing nodes that consistently lead to high-reward outcomes while gradually discouraging exploration of suboptimal branches. By aggregating reward information in this manner, the tree progressively biases future selections toward more promising regions of the molecular space, improving search efficiency and optimization performance.

The MCTS algorithm initializes with a root node defined by task-specific constraints (e.g., a scaffold structure or [BOS] token) and iteratively cycles through the four phases until a termination condition is met. In this study, the typical termination criterion is a predefined number of MCTS iterations (Iteration

20

---

Limit). After optimization, the optimal molecule is selected from the leaf nodes based on the highest cumulative reward, with optional post-processing (e.g., validity checks) to refine the output.

### Code availability
The source code of Trio is available at Github: https://github.com/SZU-ADDG/Trio.

### Data availability
The data that support the findings of this study are available from the following sources: The training dataset was derived by sampling molecules from the ZINC database, which is publicly accessible at https://zinc.docking.org/. Datasets used for evaluation were obtained from public databases. All data related to the specific targets, along with the source code, are provided in the aforementioned GitHub repository.

### References
1. Lyu, J. et al. Ultra-large library docking for discovering new chemotypes. *Nature* **566**, 224–229 (2019).
2. Li, P. et al. A deep learning approach for rational ligand generation with toxicity control via reactive building blocks. *Nature Computational Science* 1–14 (2024).
3. Zhou, G. et al. An artificial intelligence accelerated virtual screening platform for drug discovery. *Nature Communications* **15**, 7761 (2024).
4. Jeon, H. et al. Stella provides a drug design framework enabling extensive fragment-level chemical space exploration and balanced multi-parameter optimization. *Scientific Reports* **15**, 28135 (2025).
5. Cai, X., Zhang, T., Qiu, Y. & Cui, Z. Fragment-driven progressive alternating diffusion for de novo molecular design. *IEEE Transactions on Computational Biology and Bioinformatics* (2025).
6. Peng, X. et al. Pocket2mol: Efficient molecular sampling based on 3d protein pockets. In *International Conference on Machine Learning*, 17644–17655 (PMLR, 2022).
7. Zhang, O. et al. Resgen is a pocket-aware 3d molecular generation model based on parallel multiscale modelling. *Nature Machine Intelligence* **5**, 1020–1030 (2023).
8. Zhang, O. et al. Fraggen: towards 3d geometry reliable fragment-based molecular generation. *Chemical Science* **15**, 19452–19465 (2024).
9. Du, Y. et al. Machine learning-aided generative molecular design. *Nature Machine Intelligence* **6**, 589–604 (2024).
10. Guan, J. et al. 3d equivariant diffusion for target-aware molecule generation and affinity prediction. *arXiv preprint arXiv:2303.03543* (2023).

---

```markdown
11. Lin, H. et al. Diffbp: Generative diffusion of 3d molecules for target protein binding. *Chemical Science* **16**, 1417–1431 (2025).
12. Schneuing, A. et al. Structure-based drug design with equivariant diffusion models. *Nature Computational Science* **4**, 899–909 (2024).
13. Song, Y. et al. Equivariant flow matching with hybrid probability transport for 3d molecule generation. *Advances in Neural Information Processing Systems* **36**, 549–568 (2023).
14. Krishnan, A. et al. A generative deep learning approach to de novo antibiotic design. *Cell* (2025).
15. Weininger, D. Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules. *Journal of chemical information and computer sciences* **28**, 31–36 (1988).
16. Krenn, M., Häse, F., Nigam, A., Friederich, P. & Aspuru-Guzik, A. Self-referencing embedded strings (selfies): A 100% robust molecular string representation. *Machine Learning: Science and Technology* **1**, 045024 (2020).
17. Noutahi, E., Gabellini, C., Craig, M., Lim, J. S. & Tossou, P. Gotta be safe: a new framework for molecular design. *Digital Discovery* **3**, 796–804 (2024).
18. Zholus, A. et al. Bindgpt: A scalable framework for 3d molecular design via language modeling and reinforcement learning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 39, 26083–26091 (2025).
19. Wang, J. et al. 3dsmiles-gpt: 3d molecular pocket-based generation with token-only large language model. *Chemical Science* **16**, 637–648 (2025).
20. Wu, K. et al. Tamgen: drug design with target-aware molecule generation through a chemical language model. *Nature Communications* **15**, 9360 (2024).
21. Sun, K. et al. Synllama: Generating synthesizable molecules and their analogs with large language models. *ACS Central Science* (2025).
22. Zhang, K. et al. Artificial intelligence in drug development. *Nature medicine* **31**, 45–59 (2025).
23. Zitnik, M. Ai-enabled drug discovery reaches clinical milestone: Machine learning. *Nature Medicine* 1–2 (2025).
24. Rafailov, R. et al. Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems* **36**, 53728–53741 (2023).
25. Köpf, A. et al. Openassistant conversations-democratizing large language model alignment. *Advances in Neural Information Processing Systems* **36**, 47669–47681 (2023).
26. Noutahi, E., Gabellini, C., Craig, M., Lim, J. S. & Tossou, P. Gotta be safe: a new framework for molecular design. *Digital Discovery* **3**, 796–804 (2024).
```

---

```
- [27] Lee, S. et al. Genmol: A drug discovery generalist with discrete diffusion. *arXiv preprint arXiv:2501.06158* (2025).
- [28] Lee, S. et al. Molecule generation with fragment retrieval augmentation. *Advances in Neural Information Processing Systems* **37**, 132463–132490 (2024).
- [29] Jin, W., Barzilay, R. & Jaakkola, T. Junction tree variational autoencoder for molecular graph generation. In *International conference on machine learning*, 2323–2332 (PMLR, 2018).
- [30] Olivercrona, M., Blaschke, T., Engkvist, O. & Chen, H. Molecular de-novo design through deep reinforcement learning. *Journal of cheminformatics* **9**, 1–14 (2017).
- [31] Jensen, J. H. A graph-based genetic algorithm and generative model/monte carlo tree search for the exploration of chemical space. *Chemical science* **10**, 3567–3572 (2019).
- [32] Jeon, W. & Kim, D. Autonomous molecule generation using reinforcement learning and docking to develop potential novel inhibitors. *Scientific reports* **10**, 22104 (2020).
- [33] Jin, W., Barzilay, R. & Jaakkola, T. Hierarchical generation of molecular graphs using structural motifs. In *International conference on machine learning*, 4839–4848 (PMLR, 2020).
- [34] Nigam, A., Friederich, P., Krenn, M. & Aspuru-Guzik, A. Augmenting genetic algorithms with deep neural networks for exploring the chemical space. *arXiv preprint arXiv:1909.11655* (2019).
- [35] Xie, Y. et al. Mars: Markov molecular sampling for multi-objective drug discovery. *arXiv preprint arXiv:2103.10432* (2021).
- [36] Ahn, S., Kim, J., Lee, H. & Shin, J. Guiding deep molecular optimization with genetic exploration. *Advances in neural information processing systems* **33**, 12008–12021 (2020).
- [37] Jin, W., Barzilay, R. & Jaakkola, T. Multi-objective molecule generation using interpretable sub-structures. In *International conference on machine learning*, 4849–4859 (PMLR, 2020).
- [38] Yang, S., Hwang, D., Lee, S., Ryu, S. & Hwang, S. J. Hit and lead discovery with explorative rl and fragment-based molecule generation. *Advances in Neural Information Processing Systems* **34**, 7924–7936 (2021).
- [39] Kong, X., Huang, W., Tan, Z. & Liu, Y. Molecule generation by principal subgraph mining and assembling. *Advances in Neural Information Processing Systems* **35**, 2550–2563 (2022).
- [40] Lee, N. et al. Conditional graph information bottleneck for molecular relational learning. In *Inter- national Conference on Machine Learning*, 18852–18871 (PMLR, 2023).
- [41] Wang, Z. et al. Retrieval-based controllable molecule generation. In *International Conference on Learning Representations* (ICLR) 2023 (2023).

23
```

---

```markdown
[42] Kim, H., Kim, M., Choi, S. & Park, J. Genetic-guided gflownets for sample efficient molecular optimization. In *The thirty-eighth Annual Conference on Neural Information Processing Systems* (2024).

[43] Lee, S., Lee, S., Kawaguchi, K. & Hwang, S. J. Drug discovery with dynamic goal-aware fragments. In *International Conference on Machine Learning*, 26731–26751 (PMLR, 2024).

[44] Xie, Y., Xu, Z., Ma, J. & Mei, Q. How much space has been explored? measuring the chemical space covered by databases and machine-generated molecules. In *11th International Conference on Learning Representations, ICLR 2023* (2023).

[45] Irwin, J. J. & Shoichet, B. K. Zinc- a free database of commercially available compounds for virtual screening. *Journal of chemical information and modeling* **45**, 177–182 (2005).

[46] Chambers, J. et al. Unichem: a unified chemical structure cross-referencing and identifier tracking system. *Journal of cheminformatics* **5**, 3 (2013).

[47] Podda, M., Bacciu, D. & Micheli, A. A deep generative model for fragment-based molecule generation. In *International conference on artificial intelligence and statistics*, 2240–2250 (PMLR, 2020).

[48] Degen, J., Wegscheid-Gerlach, C., Zaliani, A. & Rarey, M. On the art of compiling and using’drug-like’chemical fragment spaces. *ChemMedChem* **3**, 1503 (2008).

[49] Schwaller, P. et al. Molecular transformer: a model for uncertainty-calibrated chemical reaction prediction. *ACS central science* **5**, 1572–1583 (2019).

[50] Su, J. et al. Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing* **568**, 127063 (2024).

[51] Olivecrona, M., Blaschke, T., Engkvist, O. & Chen, H. Molecular de-novo design through deep reinforcement learning. *Journal of cheminformatics* **9**, 1–14 (2017).

[52] Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347* (2017).

[53] Chaslot, G., Bakkes, S., Szita, I. & Spronck, P. Monte-carlo tree search: A new framework for game ai. In *Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, vol. 4*, 216–217 (2008).

[54] Auer, P., Cesa-Bianchi, N. & Fischer, P. Finite-time analysis of the multiarmed bandit problem. *Machine learning* **47**, 235–256 (2002).

[55] Tian, Y. et al. Toward self-improvement of llms via imagination, searching, and criticizing. *Advances in Neural Information Processing Systems* **37**, 52723–52748 (2024).

24
```

---

```
# Acknowledgements

This work was supported by the National Natural Science Foundation of China under Grant 6247617, and the Guangdong Natural Science Foundation Project under Grant 2025A1515011567.

# Author Contributions

Z.Y. and J.J. conceived the idea for Trio. J.J., R.B., and Z.Z. coordinated and supervised the project. Z.Y. designed and implemented the complete workflow and algorithms of Trio, with contributions from D.X. to the implementation. Z.Y. and D.X. conducted the experiments, analyzed the data, and drafted the manuscript. J.J., J.L., R.B., T.H., and Z.Z. provided critical feedback on the algorithm evaluation and revised the manuscript.

# Declaration of Interests

The authors declare no competing interests.

[Page 25]
```