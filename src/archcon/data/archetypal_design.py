"""Design-only helpers for the post-pretraining archetypal stage.

This module intentionally contains no IKEM training loop yet.  It makes the
future model choices explicit now, while keeping GEO pretraining archetype-free.
"""

from __future__ import annotations

from dataclasses import dataclass


AA_BASELINE = "Stadniuk baseline · contrastive side information"
AA_SHARED = "Shared-W multimodal archetypes · recommended next"
AA_CONDITIONAL = "Conditional membership prior"
AA_HYBRID = "Shared-W + contrastive hybrid"
AA_OUTCOME = "Outcome-guided archetypes · strict nested CV"
AA_DESIGN_OPTIONS = [AA_BASELINE, AA_SHARED, AA_CONDITIONAL, AA_HYBRID, AA_OUTCOME]


@dataclass(frozen=True)
class AADesign:
    label: str
    short: str
    needs_clinical: bool
    needs_outcome: bool


AA_DESIGNS = {
    AA_BASELINE: AADesign(
        AA_BASELINE,
        "Clinical similarity pulls/pushes molecular latent codes; clinical variables are not reconstructed.",
        True,
        False,
    ),
    AA_SHARED: AADesign(
        AA_SHARED,
        "One membership vector W explains both molecular latent state and clinical phenotype.",
        True,
        False,
    ),
    AA_CONDITIONAL: AADesign(
        AA_CONDITIONAL,
        "Clinical variables alter a prior/regularizer over W instead of only pairwise distances.",
        True,
        False,
    ),
    AA_HYBRID: AADesign(
        AA_HYBRID,
        "Shared-W clinical reconstruction plus a weaker contrastive geometry term.",
        True,
        False,
    ),
    AA_OUTCOME: AADesign(
        AA_OUTCOME,
        "Outcome/eGFR contributes only inside the training fold; this changes the scientific claim to prognostic archetypes.",
        True,
        True,
    ),
}


def aa_design_markdown(mode: str) -> str:
    if mode == AA_BASELINE:
        return r"""
### Baseline · side information changes geometry

Molecular expression is encoded first, then archetypal structure is imposed in latent space. Clinical features influence a supervised contrastive term:

$$
X\to z,\qquad z\approx W A_Z,
$$

$$
\mathcal L=\alpha\mathcal L_{\rm recon}+\beta\mathcal L_{\rm arch}
+\gamma\mathcal L_{\rm sep}+\eta\mathcal L_{\rm entropy}
+\delta\mathcal L_{\rm contrastive}(F,z).
$$

This reproduces the conceptual limitation we want to move beyond: **$F$ changes where samples sit, but the archetypal mixture does not explicitly explain $F$.**
"""

    if mode == AA_SHARED:
        return r"""
### Shared-W multimodal archetypes · strongest next direction

Use the **same archetypal memberships** for molecular and clinical views:

$$
z_i\approx W_i A_Z,
\qquad
F_i\approx W_i A_F.
$$

A donor is then interpreted as one mixture $W_i$ whose extreme states have both a molecular profile and a clinical profile. For mixed clinical variables:

$$
\mathcal L_F=\sum_j \lambda_j\,\mathcal L_j(F_j,\hat F_j),
$$

with Gaussian/MSE for continuous variables, Bernoulli/BCE for binary variables, categorical cross-entropy for nominal variables, ordinal loss for grades, and masking for missing values.

This is closer to **multiview archetypal analysis** than merely using clinical variables to push and pull molecular embeddings.
"""

    if mode == AA_CONDITIONAL:
        return r"""
### Conditional membership prior

Keep molecular reconstruction primary, but let clinical information define a prior or regularizer for archetypal membership:

$$
W_i=q_\psi(W\mid z_i),
\qquad
p_\omega(W\mid F_i),
$$

$$
\mathcal L_{\rm side}=D\!\left(q_\psi(W\mid z_i),\,p_\omega(W\mid F_i)\right).
$$

This makes the clinical role explicit at the **membership level**, while avoiding the stronger assumption that every clinical variable itself must be reconstructed by a convex mixture.
"""

    if mode == AA_HYBRID:
        return r"""
### Hybrid · shared archetypes + local geometry

Use shared memberships as the main semantic constraint,

$$
z\approx W A_Z,\qquad F\approx W A_F,
$$

then retain a smaller contrastive term to encourage clinically similar donors to remain locally coherent inside or between archetypal mixtures:

$$
\mathcal L=\mathcal L_{\rm recon}+\mathcal L_{\rm AA,Z}
+\lambda_F\mathcal L_{\rm AA,F}+\delta\mathcal L_{\rm contrastive}.
$$

This is more flexible, but it also introduces more objectives that can fight each other, so ablation tests will be essential.
"""

    return r"""
### Outcome-guided archetypes · separate scientific question

An outcome head could use memberships directly,

$$
\hat y_i(t)=g(W_i,t,\text{covariates}),
$$

or add an outcome loss to the latent/archetypal model. This can produce explicitly **prognostic archetypes**, but outcome information must never cross the held-out fold boundary. Hyperparameter selection also has to be nested inside training folds.

This option should remain disabled until the supervised-dataset outcome table and evaluation protocol are wired into ArchCon.
"""


def aa_research_markdown() -> str:
    return """
### 🔬 Design references and cautions

- **MIDAA (Genome Biology, 2025)** uses modality-specific encoders followed by a shared encoder/latent archetypal structure and reverses this design in the decoder. It is strong evidence that a multiview biological AA design is practical: https://link.springer.com/article/10.1186/s13059-025-03530-9
- **Archetypal SAE (ICML 2025)** constrains learned dictionary atoms to the data convex hull, an idea worth testing for more interpretable/stable extreme profiles: https://proceedings.mlr.press/v267/fel25a.html
- A **2026 ablation paper** questions whether the reported Archetypal-SAE stability advantage survives changes in initialization/metric design. We should therefore measure stability across independent seeds instead of assuming the constraint guarantees it: https://arxiv.org/abs/2606.02061

**Implementation decision:** expose these designs now, but keep GEO pretraining archetype-free. Actual AA fitting waits for supervised-dataset molecular rows plus typed clinical metadata. No archetype class labels are required for unsupervised/shared-W AA.
"""


def aa_design_svg(mode: str) -> str:
    shared = mode in {AA_SHARED, AA_HYBRID}
    conditional = mode == AA_CONDITIONAL
    outcome = mode == AA_OUTCOME
    width = 1500
    height = 500
    nodes = [
        (70, 185, 220, 120, "Molecular X", "42,917 probes"),
        (360, 185, 200, 120, "Encoder", "pretrained"),
        (640, 185, 170, 120, "z", "latent"),
        (900, 185, 170, 120, "W", "archetype mix"),
        (1160, 185, 240, 120, "A_Z", "molecular extremes"),
    ]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="aa-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="currentColor"/></marker></defs>',
        '<style>text{font-family:Inter,system-ui,sans-serif;fill:currentColor}.n{fill:transparent;stroke:currentColor;stroke-width:2.4}.a{stroke:currentColor;stroke-width:2.2;marker-end:url(#aa-arrow)}.d{stroke-dasharray:8 6}</style>',
    ]
    for x, y, w, h, title, subtitle in nodes:
        elements.append(f'<rect class="n" x="{x}" y="{y}" width="{w}" height="{h}" rx="16"/>')
        elements.append(f'<text x="{x+w/2}" y="{y+50}" text-anchor="middle" font-size="25" font-weight="750">{title}</text>')
        elements.append(f'<text x="{x+w/2}" y="{y+83}" text-anchor="middle" font-size="18">{subtitle}</text>')
    for x1, x2 in ((290, 360), (560, 640), (810, 900), (1070, 1160)):
        elements.append(f'<line class="a" x1="{x1}" y1="245" x2="{x2}" y2="245"/>')

    if shared:
        elements.extend([
            '<rect class="n" x="900" y="35" width="170" height="105" rx="16"/>',
            '<text x="985" y="78" text-anchor="middle" font-size="24" font-weight="750">A_F</text>',
            '<text x="985" y="108" text-anchor="middle" font-size="17">clinical extremes</text>',
            '<line class="a" x1="985" y1="185" x2="985" y2="140"/>',
            '<rect class="n" x="1160" y="35" width="240" height="105" rx="16"/>',
            '<text x="1280" y="78" text-anchor="middle" font-size="24" font-weight="750">Clinical F</text>',
            '<text x="1280" y="108" text-anchor="middle" font-size="17">mixed data types</text>',
            '<line class="a" x1="1070" y1="87" x2="1160" y2="87"/>',
        ])
    elif conditional:
        elements.extend([
            '<rect class="n" x="650" y="35" width="260" height="105" rx="16"/>',
            '<text x="780" y="78" text-anchor="middle" font-size="24" font-weight="750">p(W | F)</text>',
            '<text x="780" y="108" text-anchor="middle" font-size="17">clinical membership prior</text>',
            '<line class="a d" x1="865" y1="140" x2="930" y2="185"/>',
        ])
    else:
        elements.extend([
            '<rect class="n" x="620" y="35" width="310" height="105" rx="16"/>',
            '<text x="775" y="78" text-anchor="middle" font-size="23" font-weight="750">Clinical F</text>',
            '<text x="775" y="108" text-anchor="middle" font-size="17">contrastive side information</text>',
            '<line class="a d" x1="775" y1="140" x2="740" y2="185"/>',
        ])
    if outcome:
        elements.extend([
            '<rect class="n" x="900" y="355" width="260" height="105" rx="16"/>',
            '<text x="1030" y="398" text-anchor="middle" font-size="23" font-weight="750">Outcome head</text>',
            '<text x="1030" y="428" text-anchor="middle" font-size="17">strict training fold only</text>',
            '<line class="a d" x1="985" y1="305" x2="1030" y2="355"/>',
        ])
    elements.append('</svg>')
    return '<div class="architecture-scroll-shell aa-design-shell">' + ''.join(elements) + '</div>'
