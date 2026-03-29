# Wave Overtopping Prediction Using Neural Networks

This manual brings together **overtopping theory**, **dimensional
analysis**, **EurOtop context**, **neural-network methodology**,
**recent machine-learning interpretation**, and a **full operator
manual** for the current Python implementation.

## 1. Purpose, scope, and how this manual should be used

**This document has two equally important objectives.** First, it
provides a **high-detail theoretical introduction** to the prediction of
mean wave overtopping discharge using empirical reasoning, dimensionless
analysis, and neural-network or machine-learning models. Second, it
provides a **step-by-step operating manual** for the exact workflow 
consisting of **train.py** as the computational backend and
**gui.py/gui.exe** as the desktop interface.

**The manual is written for engineers, analysts, and technically
literate operators.** It assumes that the reader needs not only a
superficial explanation of where to click or what command to run, but
also a strong understanding of **what the model is predicting**, **which
physical variables control the answer**, **how uncertainty should be
interpreted**, **why extrapolation is dangerous**, and **how the
Python implementation differs from the official EurOtop ANN
tools** described in the literature.

**The practical scope is very specific.** The workflow can be
used for **single-scenario prediction**, **batch prediction**, **model
training and re-training**, **diagnostics generation**, and **range
checking against the calibration domain**. The GUI documentation
included in the GUI documentation explicitly states these intended tasks and
also makes clear that the GUI is aligned to a **two-stage backend
workflow** composed of **holdout validation** followed by **full-data
refit**. That distinction is crucial because the validation stage
estimates generalisation quality, whereas the refit stage produces the
saved production model that is later used for prediction.

**This is not a manual for all overtopping methods ever published.**
Instead, it is a manual that places the current workflow into the
broader overtopping framework represented by **EurOtop** and subsequent
neural-network and machine-learning work. In practice, that means the
text below repeatedly compares three layers of understanding:
**classical overtopping physics**, **official ANN/XGB overtopping tools
from the literature**, and **the exact custom implementation in the
project Python files**.

**How to use this manual in practice.** A first-time user should read
Sections 2 to 8 in sequence. An experienced operator who mainly needs
procedures may jump directly to Sections 9 and 10. A reviewer who needs
to verify whether a prediction is credible should focus on Sections 8,
11, and 12, plus Appendix E. A developer who intends to maintain or
extend the workflow should pay particular attention to the sections
describing **target definition**, **engineered features**, **ensemble
construction**, **holdout diagnostics**, and **full-data refit**.

> **Warning. Reading rule for safe engineering use.** Whenever the manual says **"prediction"**, read that as **"model estimate conditioned on the calibration data and on the assumptions encoded in the inputs"**. It does **not** mean physical truth. The more the input scenario departs from the training domain, the more the prediction becomes an extrapolation rather than an interpolation.

## 2. Executive technical summary

**Wave overtopping** is the mean or event-scale passage of water over
the crest of a coastal defence, seawall, dike, revetment, rubble-mound
breakwater, or vertical wall when wave run-up, impact, or splash exceeds
the available crest freeboard. In practical design and assessment, the
main scalar quantity is the **mean discharge q**, usually reported in
**m³/s/m** or **l/s/m**.

**Classical overtopping engineering** expresses the process using a
combination of **wave conditions at the structure toe**, **geometry of
the structure**, **roughness and permeability effects**, and **regime
indicators** such as the breaker parameter or surf similarity parameter.
The EurOtop formulation for smooth slopes is often written in the
generic form **$q / \sqrt{g\,H_{m0}^{3}} = a\,\exp\!\left[-\left(b\,R_c/H_{m0}\right)^{c}\right]$**, with
regime-specific expressions for breaking and non-breaking waves **(van
der Meer et al., 2018)**.

**Neural-network and machine-learning models** are attractive because
overtopping depends on a **highly nonlinear interaction** between
hydraulic forcing and geometric configuration. A sufficiently broad and
homogeneous database allows a model to learn relationships that are
difficult to represent with a single simple closed-form equation,
especially for **composite**, **bermed**, or otherwise **complex
structures** **(van Gent et al., 2007; Formentin et al., 2017; den
Bieman et al., 2021)**.

**The official EurOtop ANN** is a specific reference point. It uses
**dimensionless physically motivated inputs** and, in the 2018
description, derives uncertainty from the distribution of predictions
from **500 neural networks**, providing a **mean** and a **5%/95%
confidence interval** **(van der Meer et al., 2018)**.

**The `train.py` model in this project is not the official EurOtop ANN.** It is
a **custom bagged multilayer perceptron ensemble** implemented with
**scikit-learn MLPRegressor**, median imputation, standardisation, early
stopping, and engineered hydraulic features. It trains on **$\log_{10}(s_q)$**
and saves a **model.joblib** bundle plus **diagnostics.json** and plot
outputs. By default it uses **10 bagged MLP models**, **hidden layers
(192, 96, 48, 24)**, and a **15% holdout split**.

**A critical conceptual update must be stated explicitly.** In the
corrected revision of **train.py**, the internal target is defined as
**$s_q = q / \sqrt{g H_{m0,\mathrm{toe}}^3}$**. This is the standard
non-dimensional overtopping-discharge scaling used in EurOtop-style
practice, expressed here with the **toe-based spectral wave height**
**$H_{m0,\mathrm{toe}}$**. Therefore, when the script displays **sq**, it is now a
physically standard dimensionless discharge coefficient rather than a
non-standard custom scaling.

**The GUI is a practical engineering shell around that backend.** It
supports immediate model loading, single-case prediction, batch
prediction, model training/refresh, diagnostics inspection,
calibration-range checking, persistent defaults, and CSV export. The GUI
is designed so that a user can treat the workflow as a repeatable
desktop tool rather than as a command-line-only program.

**The safest way to use the current workflow** is: train from a
well-curated database; review diagnostics; check whether the target case
is inside the observed training range; run prediction; inspect the
spread indicators **P05, P50, P95** and any range warning; and then
decide whether the result is acceptable as a preliminary estimate or
whether more rigorous project-specific analysis is required.

High-level comparison of conceptual layers used throughout this manual

- **Classical overtopping physics**
  - **What it does**: Explains **why** overtopping occurs and what variables dominate it.
  - **Why it matters operationally**: Provides engineering intuition and guards against blind trust in black-box predictions.

- **Empirical formulae**
  - **What it does**: Give direct equations for representative structure classes.
  - **Why it matters operationally**: Useful as a cross-check and as a baseline for reasonableness.

- **Official EurOtop ANN/XGB tools**
  - **What it does**: Demonstrate how data-driven methods were formalised in practice.
  - **Why it matters operationally**: Provide context, precedent, and cautionary limits.

- **Uploaded custom workflow**
  - **What it does**: Implements a specific bagged MLP model and a desktop GUI.
  - **Why it matters operationally**: This is the exact workflow the user will actually run.

## 3. Wave overtopping fundamentals

**Wave overtopping is not a steady fluid discharge phenomenon.** The
underlying hydrodynamic process is strongly unsteady: individual waves
run up the structure, some fail to reach the crest, some barely exceed
it, and some produce large bursts of water over the crest. The commonly
reported discharge **q** is therefore a **mean discharge over time and
per unit width**, not a literal continuous sheet of water of constant
thickness. EurOtop explicitly emphasises that overtopping can arise as
**green water**, **impact-driven throw over the crest**, and
**splash-related processes**, although practical prediction formulae
usually compress that complexity into a mean discharge variable **(van
der Meer et al., 2018)**.

**Why mean discharge became the central overtopping variable.** Mean
discharge is comparatively easy to measure in laboratory flumes and
basins, easier to store in databases, and easier to use in design
tolerability criteria than full time histories of overtopping
velocities, thicknesses, and individual volumes. That historical
measurability is one reason why the overtopping literature and later
data-driven models are built so heavily around **q**.

**The physical mechanism is a balance between forcing and resistance.**
The forcing side includes water level, significant wave height, wave
period, obliquity, breaking regime, and depth-limited transformation
before the structure. The resistance side includes crest freeboard,
slope shape, toe and berm configuration, crest width, roughness,
permeability, and the ability of the structure to dissipate or reflect
wave energy. Overtopping increases when forcing rises or resistance
falls. That sounds trivial, but it is the organising principle behind
nearly all empirical equations and nearly all engineered
machine-learning features.

**Three principal structural families dominate the engineering
framework.** First, **sloping dikes and embankment seawalls**, where
run-up on the slope is central. Second, **armoured rubble-mound
structures**, where permeability, roughness, armour configuration, and
berm effects become more influential. Third, **vertical or very steep
walls**, where non-impulsive and impulsive impact regimes must be
distinguished and where foreshore effects can drastically change the
overtopping mechanism **(van der Meer et al., 2018)**.

**The concept of "the toe of the structure" is operationally
essential.** Overtopping inputs are not defined by offshore wave
conditions alone. They must be defined at the **toe region** or at the
point where the structure sees the transformed waves after shoaling,
depth limitation, and breaking. This matters because the current
workflow, like the broader EurOtop approach, uses **$H_{m0,\mathrm{toe}}$** and
**$T_{m-1,0,\mathrm{toe}}$** as key inputs rather than offshore-only descriptors.

**A mean overtopping value is meaningful only relative to use and
consequence.** A discharge that may be acceptable for a rough breakwater
with no public access may be unacceptable for a promenade, railway,
road, utility corridor, or flood defence line. In other words,
overtopping prediction is only half of the engineering question. The
other half is **tolerability**: what discharge is acceptable for people,
vehicles, assets, and the structural system itself **(van der Meer et
al., 2018)**.

Principal structure families and overtopping interpretation

- **Smooth sloping dike / seawall**
  - **Typical dominant controls**: Run-up, freeboard, breaker regime, obliquity, roughness
  - **Machine-learning implication**: Relatively clear physics; empirical formulae are useful and should often be checked against ML estimates.

- **Rubble-mound / armoured slope**
  - **Typical dominant controls**: Permeability, roughness, berm effects, crest geometry, armour characteristics
  - **Machine-learning implication**: Data-driven methods are valuable because multiple interacting effects are difficult to collapse into one short equation.

- **Vertical / steep wall**
  - **Typical dominant controls**: Foreshore influence, impulsive versus non-impulsive attack, freeboard, relative depth
  - **Machine-learning implication**: Regime classification becomes critical; inappropriate extrapolation can be highly misleading.

## 4. Key hydraulic and structural variables

**This section is foundational.** A neural-network model can only be
interpreted correctly if each input variable is understood physically.
The project scripts use a compact set of base variables and then derive
additional dimensionless and trigonometric features from them. That
means the reader must understand both the **engineering meaning of the
base inputs** and the **mathematical role of the engineered variables**.

**Significant wave height $H_{m0}$.** In the EurOtop framework the preferred
wave height for overtopping is the **spectral significant wave height
$H_{m0}$**, often written as **$H_{m0} = 4\sqrt{m_0}$**, where **$m_0$** is the zeroth
spectral moment. The manual explicitly distinguishes $H_{m0}$ from $H_{1/3}$,
noting that differences can arise in shallow water and that the
overtopping equations are generally built around $H_{m0}$ at the toe **(van
der Meer et al., 2018)**.

**Spectral period $T_{m-1,0}$.** The preferred period in much overtopping
work is the spectral period **$T_{m-1,0} = m_{-1} / m_0$**. This period gives
more weight to the longer wave components than a simple mean period and
is therefore well suited to run-up and overtopping work. The project
scripts accept **$T_{m-1,0,\mathrm{toe}}$** directly as an input, reflecting this
established practice.

**Deep-water wavelength based on $T_{m-1,0}$.** A common derived quantity is
**$L_{m-1,0} = g\,T_{m-1,0}^{2} / (2\pi)$**. The `train.py` script computes
**$L_{m-1,0,\mathrm{toe}}$** using exactly this form. This conversion is not a minor
detail: many dimensionless overtopping inputs use a **length
normalization by wavelength** rather than by geometric length in metres
alone.

**Wave steepness.** Wave steepness is classically a ratio of wave height
to wavelength. The script computes a form of **`wave_steepness` $= H_{m0} / L_0$**. That in turn feeds the square-root steepness and
breaker-parameter-style features used by the model.

**Breaker parameter / surf similarity $\xi$.** The EurOtop definition is
commonly written as **$\xi_{m-1,0} = \tan(\alpha) / \sqrt{H_{m0} / L_{m-1,0}}$**. It
is a compact regime indicator linking slope and wave steepness. Low
values correspond more to breaking or plunging behaviour, while higher
values correspond more to surging or non-breaking behaviour. The
the script creates **`xi_m10_lower`** and **`xi_m10_upper`** by
combining the lower and upper slope cotangents with square-root wave
steepness.

**Freeboard and crest geometry.** **$R_c$** is crest freeboard; **$A_c$** is
armour crest freeboard; **$G_c$** is crest width. These variables matter
because overtopping is controlled by how much vertical and horizontal
crest resistance exists after run-up has developed. The current
workflow uses both raw values and normalized ratios such as **$R_c/H_{m0}$**,
**$A_c/H_{m0}$**, **$G_c/L_{m-1,0}$**, and **$G_c/H_{m0}$**.

**Toe and berm geometry.** **$h_t$** is toe water depth, **$B_t$** is toe
width, **$B$** is berm width, and **$h_b$** is berm water depth or berm level
descriptor according to the data convention. In the current workflow,
these are not merely retained as raw numbers; they are intentionally
normalized by **$H_{m0}$** or **$L_{m-1,0}$** so the model can learn more
transferable patterns across scales.

**Obliquity $\beta$.** The scripts accept **$\beta$** in degrees and also
derive **$|\beta|$**, **$\cos(\beta)$**, and **$|\sin(\beta)|$**. This is
a good example of engineered learning design: the raw angle alone may
not be the most convenient numerical representation for a multilayer
perceptron, whereas trigonometric transforms encode directional
behaviour more smoothly.

**Roughness/permeability factor $g_f$.** The overtopping literature often
uses influence factors to account for roughness and permeability
effects. The script retains **$g_f$** as a raw base input and also
creates the interaction feature **`gf_cos_beta`**, which effectively
couples roughness/permeability behaviour with the directional attack
angle.

**Slope representation through cotangents.** The project files use
**cotad** and **cotau**, representing the lower and upper slope
cotangents. Since many overtopping formulae are written in terms of
**$\tan(\alpha)$**, the script internally reconstructs inverse cotangents
using safe division. This allows it to form breaker-like features and
other ratio features without requiring the user to enter slope angles
directly.

Base input variables used directly by the current workflow

- **m**
  - **Meaning**: Foreshore slope cotangent in the CLI help of train.py
  - **Role in script**: Base input
  - **Operational significance**: Controls antecedent geometry and near-structure wave transformation context.

- **$b$ / $\beta$**
  - **Meaning**: Wave obliquity in degrees
  - **Role in script**: Base input
  - **Operational significance**: Directional attack angle; also transformed to cos and sin-based features.

- **h**
  - **Meaning**: Water depth at the toe/front of the structure
  - **Role in script**: Base input
  - **Operational significance**: Sets local depth context and depth-limited behaviour.

- **$H_{m0,\mathrm{toe}}$**
  - **Meaning**: Spectral significant wave height at the toe
  - **Role in script**: Base input
  - **Operational significance**: Primary wave-height scaling variable.

- **$T_{m-1,0,\mathrm{toe}}$**
  - **Meaning**: Spectral mean period at the toe
  - **Role in script**: Base input
  - **Operational significance**: Period used to derive wavelength and regime indicators.

- **ht**
  - **Meaning**: Toe water depth
  - **Role in script**: Base input
  - **Operational significance**: Captures toe submergence and local geometric control.

- **Bt**
  - **Meaning**: Toe width
  - **Role in script**: Base input
  - **Operational significance**: Represents toe-geometry length scale.

- **gf**
  - **Meaning**: Roughness/permeability factor
  - **Role in script**: Base input
  - **Operational significance**: Dissipation and roughness control.

- **cotad**
  - **Meaning**: Lower slope cotangent
  - **Role in script**: Base input
  - **Operational significance**: Used to derive lower-slope breaker parameter.

- **cotau**
  - **Meaning**: Upper slope cotangent
  - **Role in script**: Base input
  - **Operational significance**: Used to derive upper-slope breaker parameter.

- **B**
  - **Meaning**: Berm width
  - **Role in script**: Base input
  - **Operational significance**: Horizontal berm control.

- **hb**
  - **Meaning**: Berm water depth / berm level descriptor
  - **Role in script**: Base input
  - **Operational significance**: Vertical berm control in the data model.

- **Rc**
  - **Meaning**: Crest freeboard
  - **Role in script**: Base input
  - **Operational significance**: First-order resistance to overtopping.

- **Ac**
  - **Meaning**: Armour crest freeboard
  - **Role in script**: Base input
  - **Operational significance**: Secondary crest elevation descriptor.

- **Gc**
  - **Meaning**: Crest width
  - **Role in script**: Base input
  - **Operational significance**: Crest length scale relevant to post-run-up passage.

## 5. Core overtopping formulae used as theoretical context

**The `train.py` script does not directly calculate overtopping from
closed-form empirical equations.** It is a learned model. However, a
serious manual must still present the main formulae that define the
overtopping field, because these formulae explain **why the chosen
features make physical sense** and provide a **reasonableness
benchmark** for the outputs.

**Generic overtopping form.** EurOtop presents a generic overtopping
discharge relation in a Weibull-like shape, commonly written as **$q /
\sqrt{g\,H_{m0}^{3}} = a\,\exp\!\left[-\left(b\,R_c/H_{m0}\right)^{c}\right]$** for **$R_c \geq 0$**. This compact
expression says that overtopping decays quickly with increasing relative
freeboard, but not necessarily with a simple straight exponential law in
all regimes **(van der Meer et al., 2018)**.

**Breaking and non-breaking smooth-slope formulae.** For smooth slopes,
EurOtop gives separate formulations for breaking and non-breaking wave
attack. These equations are among the most influential reference
expressions in the field and are repeatedly used as comparison baselines
in later machine-learning studies.

**Classical smooth-slope breaking-wave form (mean-value approach):**

$$
\frac{q}{\sqrt{g \, H_{m0}^{3}}}
= 0.023 \, \frac{\sqrt{\tan(\alpha)}}{\gamma_b \, \xi_{m-1,0}}
\exp\!\left[-\left(
\frac{2.7 \, R_c}{\xi_{m-1,0} \, H_{m0} \, \gamma_b \, \gamma_f \, \gamma_{\beta} \, \gamma_v}
\right)^{1.3}\right]
$$

**Classical smooth-slope non-breaking / upper-limit form:**

$$
\frac{q}{\sqrt{g \, H_{m0}^{3}}}
= 0.09 \, \exp\!\left[-\left(
\frac{1.5 \, R_c}{H_{m0} \, \gamma_f \, \gamma_{\beta} \, \gamma_*}
\right)^{1.3}\right]
$$

**Interpretation.** These expressions say that overtopping increases
with larger incident waves, lower freeboard, smoother surfaces, more
favorable direction of attack, and more energetic run-up regimes. They
also show why a learning model that receives **$R_c/H_{m0}$**, **roughness
descriptors**, **obliquity**, **wavelength-based ratios**, and **slope
parameters** is physically sensible rather than arbitrary.

**Breaker parameter definition.** A key regime variable is the surf
similarity or breaker parameter:

$$
\xi_{m-1,0} = \frac{\tan(\alpha)}{\sqrt{H_{m0} / L_{m-1,0}}}
$$

with

$$
L_{m-1,0} = \frac{g \, T_{m-1,0}^{2}}{2 \, \pi}
$$

**This parameter is central because it condenses both wave steepness and
slope steepness.** In `train.py`, the feature names
**`xi_m10_lower`** and **`xi_m10_upper`** are clear descendants of this
logic, computed separately for the lower and upper slopes from **cotad**
and **cotau**.

**Representative vertical-wall context.** For vertical and steep walls,
the overtopping regime must distinguish **non-impulsive** from
**impulsive** attack. Under non-impulsive conditions, the discharge can
be described by an exponential freeboard decay. Under impulsive
conditions, especially at higher relative freeboard, a power-law form
appears because large, violent impact-related overtopping can persist
even for high freeboards **(van der Meer et al., 2018)**.

**Representative vertical-wall context formulas from EurOtop:**

**Non-impulsive design / assessment form:**

$$
\frac{q}{\sqrt{g \, H_{m0}^{3}}} = 0.062 \, \exp\!\left(-2.61 \, \frac{R_c}{H_{m0}}\right)
$$

**Impulsive high-freeboard mean-value form:**

$$
\frac{q}{\sqrt{g \, H_{m0}^{3}}}
= 0.0014 \, \left(\frac{H_{m0}}{h \, s_{m-1,0}}\right)^{0.5}
\left(\frac{R_c}{H_{m0}}\right)^{-3}
$$

valid for $R_c / H_{m0} \geq 1.35$.

**Why these equations still matter in a machine-learning manual.** Even
if the final predictor is a neural network, these equations tell the
user what trends should usually be expected: **increasing freeboard
should generally reduce overtopping**, **smoother or less dissipative
conditions should generally increase it**, **different structural
families may require different regime logic**, and **extreme geometries
deserve caution**. Whenever an ML prediction contradicts those
first-order expectations without a convincing data-based explanation,
the result deserves scrutiny.

Formula roles in the present manual

- **$q / \sqrt{g\,H_{m0}^{3}}$**
  - **Why it appears here**: Classical dimensionless overtopping discharge
  - **How it connects to the project Python model**: Benchmark for physical interpretation; this is now the same square-root scaling used internally in train.py.

- **$L_{m-1,0} = g\,T^{2} / (2\pi)$**
  - **Why it appears here**: Toe-based wavelength scaling
  - **How it connects to the project Python model**: Computed directly in train.py as **$L_{m-1,0,\mathrm{toe}}$**.

- **$\xi_{m-1,0}$**
  - **Why it appears here**: Breaker / regime indicator
  - **How it connects to the project Python model**: Approximated by engineered lower and upper slope features.

- **$R_c/H_{m0}$, $A_c/H_{m0}$, $G_c/L$**
  - **Why it appears here**: Relative freeboard and crest ratios
  - **How it connects to the project Python model**: Computed directly as engineered ratios in train.py.

- **Roughness and obliquity factors**
  - **Why it appears here**: Capture dissipation and direction effects
  - **How it connects to the project Python model**: Represented by **$g_f$**, **$\beta$**, **`cos_beta`**, **`sin_beta_abs`**, and **`gf_cos_beta`**.

## 6. Why neural networks and machine learning are used for overtopping

**Overtopping is a classic nonlinear multivariable engineering
problem.** The response is not governed by a single independent
variable. Instead, it emerges from a coupled set of geometric and
hydraulic descriptors: wave height, period, local depth, slope, toe and
berm geometry, crest configuration, roughness/permeability, and angle of
attack. Many of these variables interact multiplicatively rather than
additively. For that reason, overtopping has been a natural candidate
for data-driven models for many years.

**The historical rationale is simple.** Empirical formulae are excellent
when the structure family is simple and the assumptions behind the
formula are respected. But real structures are often **composite**,
**bermed**, **multi-slope**, or otherwise too complicated to compress
cleanly into one short equation with a few coefficients. Neural networks
were introduced to learn from large overtopping databases and thereby
interpolate across a broader space of geometries than traditional
formulae handle comfortably **(van Gent et al., 2007; Verhaeghe et al.,
2008; Formentin et al., 2017)**.

**A machine-learning model does not create physics from nothing.** It
maps patterns already present in the database. Therefore its usefulness
depends on at least four conditions. First, the database must be large
enough. Second, the database must be reasonably consistent and
homogeneous. Third, the inputs must encode the physically meaningful
variables. Fourth, the target case should lie near the domain
represented by the training data. If any of these fail, model output
quality deteriorates quickly.

**The EurOtop manual itself explicitly recognises a place for ANN
prediction.** The manual describes the ANN as particularly recommended
for **complicated structure geometries and variable wave conditions**,
while also stressing the need for a sufficiently wide and homogeneous
database **(van der Meer et al., 2018)**. That is a balanced position:
ANN is neither magic nor a gimmick; it is a practical surrogate model
whose legitimacy depends on database quality and engineering discipline.

**Recent overtopping ML literature has broadened beyond ANN.** Later
work has compared or used support vector machines, gradient-boosted
trees, random forests, Gaussian process regression, and deep learning
methods. Some studies report better benchmark performance than older ANN
formulations for particular filtered datasets or structural classes, but
the broader lesson is not that one algorithm is universally best. The
lesson is that **feature design**, **training data filtration**,
**hyperparameter tuning**, **interpretability**, and **uncertainty
awareness** matter as much as the label attached to the algorithm **(den
Bieman et al., 2021; Alqahtani et al., 2023; Elbisy, 2024; Habib et al.,
2025)**.

**Advantages of data-driven overtopping models.** They can be fast,
scalable, and convenient for repeated screening. They are well suited to
batch evaluation, sensitivity studies, preliminary dimensioning, and
rapid comparison of alternatives. Once trained, they can return
estimates almost instantly for many scenarios.

**Limitations that never disappear.** They can reproduce dataset biases.
They are weaker under extrapolation. They may be less interpretable than
direct formulae. They do not replace project-specific physical modelling
when the geometry is exceptional or when safety consequences are high.
They are also limited by the representation of extreme events in the
training data; recent work highlights that high overtopping rates are
often underrepresented, which can degrade performance in the very region
that matters most for safety **(Habib et al., 2025)**.

**The most professional stance is therefore hybrid.** Use empirical
equations to understand trend direction and order of magnitude. Use the
neural-network or machine-learning tool for rapid multi-parameter
interpolation. Use diagnostics and range checks to determine whether the
estimate is comfortably inside the learned domain. Then decide whether
the output is adequate as a first estimate or whether more rigorous
follow-up is mandatory.

Strengths and weaknesses of ML overtopping models

- **Speed**
  - **Strength**: Very fast after training
  - **Weakness / caution**: Can encourage overuse without adequate checking.

- **Complex geometry**
  - **Strength**: Can learn interactions difficult to compress into one equation
  - **Weakness / caution**: Still limited to patterns actually represented in training data.

- **Batch operation**
  - **Strength**: Excellent for many scenarios and sensitivity studies
  - **Weakness / caution**: May hide individual outlier cases unless warnings are reviewed carefully.

- **Interpretability**
  - **Strength**: Can be improved with physically designed features and post-analysis
  - **Weakness / caution**: Often weaker than direct closed-form equations.

- **Uncertainty**
  - **Strength**: Ensembles and percentiles can express spread
  - **Weakness / caution**: Spread is not the same as full epistemic certainty.

- **Design-stage use**
  - **Strength**: Useful for conceptual design and screening
  - **Weakness / caution**: Not sufficient alone for final high-consequence design.

## 7. Official EurOtop ANN versus the current custom Python workflow

**This distinction is one of the most important messages in the entire
document.** Users often hear the phrase **"overtopping neural network"**
and assume that every modern Python implementation is simply a direct
copy of the official EurOtop ANN. That is not true here. The current
code is a **custom implementation inspired by the overtopping ML
tradition**, but it is **not** the official EurOtop ANN tool as
documented in the EurOtop manual.

**Official EurOtop ANN characteristics.** EurOtop describes a tool based
on **dimensionless physically motivated inputs**. It highlights a
network architecture with **14 dimensionless input parameters**, a
hidden layer of **20 neurons and 1 bias**, and output-specific networks
for overtopping discharge, reflection, and transmission. The manual
further states that the overtopping prediction tool is based on **500
neural networks**, whose distribution of outputs yields the mean
estimate and the **5%/95% confidence interval** **(van der Meer et al.,
2018)**.

**Current `train.py` characteristics.** The current
implementation is built with **scikit-learn**. It constructs a
**Pipeline** consisting of **SimpleImputer(strategy='median')**,
**StandardScaler**, and **MLPRegressor**. The saved metadata explicitly
identifies the model type as a **"Bagged MLPRegressor ensemble on
$\log_{10}(s_q)$ with engineered hydraulic features"**. The default
configuration uses **50 models**, not 500, and the default hidden layers
are **(192, 96, 48, 24)** rather than a single 20-neuron hidden layer.

**Official target normalization versus the project target normalization.**
The official EurOtop discourse typically centres on the classical
dimensionless discharge **$Q_{\mathrm{classical}} = q / \sqrt{g H^3}$**. In the
corrected script, the internal target is now
**$s_q = q / \sqrt{g H_{m0,\mathrm{toe}}^3}$**. That means the script is now
aligned with the classical square-root scaling, while still applying it
with **toe-based wave conditions** and still fitting
**$\log_{10}(s_q)$** after floor clipping. The model therefore remains a
custom ML implementation, but its target normalisation is now directly
consistent with standard overtopping practice.

**Official inputs versus the project inputs.** The official EurOtop ANN is
explicitly based on dimensionless inputs such as **$H_{m0}/L$**, **$h/L$**,
**$h_t/H_{m0}$**, **$B_t/L$**, **$B/L$**, **$R_c/H_{m0}$**, **$A_c/H_{m0}$**, **$G_c/L$**, slope
descriptors, and roughness/permeability descriptors. The current
train.py starts from a set of **15 base dimensional inputs** and then
internally engineers dimensionless ratios and other transforms. In
practice, that means the custom script still respects the physical logic
of overtopping ML, but it does so through a different implementation
path.

**Official tool uncertainty versus the project tool uncertainty.** EurOtop
speaks about the distribution created by 500 networks and uses that to
produce a 90% confidence interval. The current script produces spread
indicators from its own bagged ensemble and reports **P05**, **P50**,
and **P95** from the distribution of ensemble predictions. These are
useful spread descriptors, but they should be described carefully as
**ensemble percentiles in this custom workflow**, not as the official
EurOtop ANN confidence methodology.

**Practical conclusion.** The current workflow should be interpreted as
a **custom overtopping neural-network estimator with strong
EurOtop-style physical feature engineering**, not as a literal
reproduction of the official EurOtop ANN executable. This is not a
weakness. It is simply a matter of being exact about what tool is
actually being used.

Official EurOtop ANN input logic as described in the EurOtop
manual

- **$H_{m0,t} / L_{m-1,0,t}$**: Wave steepness / breaking context
- **$\beta$ [rad]**: Wave obliquity
- **$h / L_{m-1,0,t}$**: Shoaling / depth context
- **$h_t / H_{m0,t}$**: Toe submergence
- **$B_t / L_{m-1,0,t}$**: Toe width effect
- **$d_b / H_{m0,t}$**: Berm level effect
- **$B / L_{m-1,0,t}$**: Berm width effect
- **$R_c / H_{m0,t}$**: Relative crest height
- **$A_c / H_{m0,t}$**: Relative armour crest height
- **$G_c / L_{m-1,0,t}$**: Crest width effect
- **$\cot \alpha_d$**: Downstream slope
- **$\cot \alpha_{\mathrm{incl}}$**: Average run-up/down slope
- **$D / H_{m0,t}$**: Structure permeability / roughness indicator
- **$\gamma_f$**: Roughness/permeability dissipation factor

At-a-glance comparison

- **Model family**
  - **Reference tool**: Official EurOtop ANN
  - **Uploaded tool**: Custom bagged MLPRegressor ensemble

- **Input philosophy**
  - **Reference tool**: Dimensionless physically designed inputs supplied to ANN
  - **Uploaded tool**: Dimensional base inputs supplied by user, then transformed into engineered features

- **Number of networks**
  - **Reference tool**: 500 networks for overtopping spread description
  - **Uploaded tool**: 50 bagged MLP models by default

- **Representative hidden structure**
  - **Reference tool**: 14 inputs -\> 20 hidden neurons -\> 1 output
  - **Uploaded tool**: Hidden layers (192, 96, 48, 24) by default

- **Target emphasis**
  - **Reference tool**: Classical ANN overtopping prediction context
  - **Uploaded tool**: $\log_{10}(s_q)$ with $s_q = q / (g H_{m0,\mathrm{toe}}^3)$

- **Spread outputs**
  - **Reference tool**: Mean with 5% and 95% exceedance / confidence outputs
  - **Uploaded tool**: Mean plus P05, P50, P95 from the custom ensemble

- **Implementation form**
  - **Reference tool**: Dedicated overtopping ANN tool
  - **Uploaded tool**: Python scikit-learn backend plus Tkinter GUI

## 8. Detailed theory of the `train.py` model

**This section describes the project backend exactly as an engineering
model, not just as software.** The goal is to show the chain from **raw
database rows** to **saved model bundle** to **predicted overtopping
discharge**.

### 8.1 Computational objective

**The backend predicts mean overtopping for arbitrary user scenarios.**
Operationally, the final user-facing quantities are **q in l/s/m** and
**q in m³/s/m**, together with ensemble percentiles and range warnings.
Internally, however, the model is trained on a custom non-dimensional
target **sq** and specifically on **$\log_{10}(s_q)$**.

**The command-line parser itself states the target definition.** The
script description in the parser is **"Neural-network predictor for
overtopping discharge using adimensional $s_q = q / \sqrt{g\,H_{m0,\mathrm{toe}}^{3}}$"**.
That single line deserves serious emphasis, because it defines the
entire numerical target space of the model.

**Why log-transform the target.** Overtopping discharges span many
orders of magnitude, from nearly zero overtopping to strongly
overtopping cases. Training directly on raw q often leads to a model
dominated by the largest values. Training on a logarithm of a
dimensionless target compresses the dynamic range and generally
stabilises the regression problem. The implementation handles
zero and near-zero values by imposing a floor before the logarithm is
taken.

### 8.2 Training data filtration and target construction

**The database is first reduced to the required columns.** The backend
requires the 15 base feature columns and the measured discharge column
**q**. Any database row missing these critical requirements cannot
participate in training.

**The script explicitly filters for valid $q$ and valid $H_{m0,\mathrm{toe}}$.** It
retains rows where **$q$ is not null**, **$H_{m0,\mathrm{toe}}$ is not null**, **$H_{m0,\mathrm{toe}}$
is finite**, and **$H_{m0,\mathrm{toe}} > 0$**. Then it clips **q** at a lower bound
of zero before constructing the target. This means negative measured
discharges, if any were present, are not used as negative values in the
learning problem.

**Target construction in the script is exact and simple.** For each
valid row, the scale factor is **$\sqrt{g\,H_{m0,\mathrm{toe}}^{3}}$**. The raw target is
**$s_q = q / \sqrt{g\,H_{m0,\mathrm{toe}}^{3}}$**. Non-finite target values are discarded.
The stored modelling target **sq_train** is then produced by clipping
**sq** at the configurable floor **sq_floor**, which defaults to
**0.01**. The regression target actually passed to the MLP ensemble is
**$y_{\log s_q} = \log_{10}(s_{q,\mathrm{train}})$**.

**Implication of the floor.** The floor is not a physical correction to
the database. It is a training stabilisation device. Cases with
extremely low or zero overtopping are compressed into the lower tail at
**sq_floor** in log-space for the purpose of fitting. In practice, this
means the model is explicitly prioritised as a stable regression tool
over many orders of magnitude, rather than a literal zero-detection
device.

**Core target construction used in `train.py`:**

$$
\begin{aligned}
\text{scale} &= \sqrt{g \, H_{m0,\mathrm{toe}}^{3}} \\
 s_q &= \frac{q}{\text{scale}} \\
 s_q &= \max(s_q, 0) \\
 s_{q,\mathrm{train}} &= \max(s_q, s_{q,\mathrm{floor}}) \\
 y &= \log_{10}(s_{q,\mathrm{train}})
\end{aligned}
$$

### 8.3 Base features and engineered features

**The model does not rely only on the 15 base inputs.** It
deliberately engineers additional features so the neural network
receives ratios and transforms that are closer to the established
overtopping physics. This is one of the strongest technical choices in
the script. It allows the model to work not only with raw dimensions in
metres and seconds, but also with physically interpretable combinations.

**Toe wavelength and wave steepness are explicitly computed.** From
**$T_{m-1,0,\mathrm{toe}}$**, the script constructs **$L_{m-1,0,\mathrm{toe}} = g\,T^{2} / (2\pi)$**
and a corresponding **$L_{0,\mathrm{toe}}$** using the same numerical form. It then
computes **`wave_steepness` $= H_{m0} / L_0$** and **`sqrt_wave_steepness`**,
which are used to build breaker-style regime features.

**Slope cotangents are converted back to tangents where needed.** Since
many classical overtopping expressions work with **$\tan(\alpha)$** rather
than **$\cot(\alpha)$**, the script computes inverse cotangent values using
safe division. Those are then combined with square-root wave steepness
to create **`xi_m10_lower`** and **`xi_m10_upper`**.

**The rest of the feature engineering is mostly dimensionless ratio
building.** This is very much in the spirit of EurOtop-style scaling.
Ratios by **$H_{m0}$** express relative heights and widths compared with wave
height. Ratios by **$L_{m-1,0}$** express relative geometric scales compared
with wavelength. Additional interaction terms encode directional effects
and combined hydraulic-roughness effects.

Engineered features created inside train.py

- **$L_{m-1,0,\mathrm{toe}}$**
  - **Computed form**: $g\,T_{m-1,0,\mathrm{toe}}^{2} / (2\pi)$
  - **Physical purpose**: Toe-based wavelength scale

- **$L_{0,\mathrm{toe}}$**
  - **Computed form**: $g\,T_{m-1,0,\mathrm{toe}}^{2} / (2\pi)$
  - **Physical purpose**: Auxiliary wavelength used for wave steepness in the script

- **`wave_steepness`**
  - **Computed form**: $H_{m0,\mathrm{toe}} / L_{0,\mathrm{toe}}$
  - **Physical purpose**: Compact forcing descriptor

- **`sqrt_wave_steepness`**
  - **Computed form**: $\sqrt{\max(\mathrm{wave\_steepness}, 0)}$
  - **Physical purpose**: Used in breaker-style features

- **`xi_m10_lower`**
  - **Computed form**: $\tan(\alpha_d) / \mathrm{sqrt\_wave\_steepness}$
  - **Physical purpose**: Lower-slope regime indicator

- **`xi_m10_upper`**
  - **Computed form**: $\tan(\alpha_u) / \mathrm{sqrt\_wave\_steepness}$
  - **Physical purpose**: Upper-slope regime indicator

- **h_over_Lm10_toe**
  - **Computed form**: $h / L_{m-1,0,\mathrm{toe}}$
  - **Physical purpose**: Depth-to-wavelength ratio

- **Hm0_toe_over_Lm10_toe**
  - **Computed form**: $H_{m0,\mathrm{toe}} / L_{m-1,0,\mathrm{toe}}$
  - **Physical purpose**: Wave steepness-like scaling

- **ht_over_Hm0_toe**
  - **Computed form**: $h_t / H_{m0,\mathrm{toe}}$
  - **Physical purpose**: Toe submergence ratio

- **Bt_over_Lm10_toe**
  - **Computed form**: $B_t / L_{m-1,0,\mathrm{toe}}$
  - **Physical purpose**: Toe width ratio by wavelength

- **B_over_Lm10_toe**
  - **Computed form**: $B / L_{m-1,0,\mathrm{toe}}$
  - **Physical purpose**: Berm width ratio by wavelength

- **hb_over_Hm0_toe**
  - **Computed form**: $h_b / H_{m0,\mathrm{toe}}$
  - **Physical purpose**: Berm vertical ratio

- **Rc_over_Hm0_toe**
  - **Computed form**: $R_c / H_{m0,\mathrm{toe}}$
  - **Physical purpose**: Classical relative freeboard

- **Ac_over_Hm0_toe**
  - **Computed form**: $A_c / H_{m0,\mathrm{toe}}$
  - **Physical purpose**: Armour freeboard ratio

- **Gc_over_Lm10_toe**
  - **Computed form**: $G_c / L_{m-1,0,\mathrm{toe}}$
  - **Physical purpose**: Crest width ratio by wavelength

- **h_over_Hm0_toe**
  - **Computed form**: $h / H_{m0,\mathrm{toe}}$
  - **Physical purpose**: Depth-to-height ratio

- **Bt_over_Hm0_toe**
  - **Computed form**: $B_t / H_{m0,\mathrm{toe}}$
  - **Physical purpose**: Toe width ratio by wave height

- **B_over_Hm0_toe**
  - **Computed form**: $B / H_{m0,\mathrm{toe}}$
  - **Physical purpose**: Berm width ratio by wave height

- **Gc_over_Hm0_toe**
  - **Computed form**: $G_c / H_{m0,\mathrm{toe}}$
  - **Physical purpose**: Crest width ratio by wave height

- **freeboard_sum_over_Hm0_toe**
  - **Computed form**: $(R_c + A_c) / H_{m0,\mathrm{toe}}$
  - **Physical purpose**: Combined crest-elevation descriptor

- **Rc_over_Ac**
  - **Computed form**: $R_c / A_c$
  - **Physical purpose**: Relative crest partitioning

- **Bt_over_B**
  - **Computed form**: $B_t / B$
  - **Physical purpose**: Relative toe/berm width balance

- **`beta_abs`**
  - **Computed form**: $|\beta|$
  - **Physical purpose**: Angle magnitude

- **`cos_beta`**
  - **Computed form**: $\cos(\beta)$
  - **Physical purpose**: Directional effect

- **`sin_beta_abs`**
  - **Computed form**: $|\sin(\beta)|$
  - **Physical purpose**: Directional effect

- **`gf_cos_beta`**
  - **Computed form**: $g_f\,\cos(\beta)$
  - **Physical purpose**: Combined roughness-direction interaction

### 8.4 Learning architecture and ensemble design

**The backend uses a scikit-learn Pipeline.** Each model in the ensemble
is a pipeline with three stages: **median imputation**,
**standardisation**, and **MLPRegressor**. This is a sensible
configuration for tabular engineering data. Median imputation protects
the model from sporadic missing values. Standardisation prevents
large-magnitude variables from numerically dominating the optimisation.
The MLP then learns a nonlinear regression map in the transformed
feature space.

**The individual neural network is a multilayer perceptron with ReLU
activation and Adam optimisation.** In the current defaults, the
hidden-layer structure is **(192, 96, 48, 24)**. Early stopping is
activated, using a validation fraction of **0.12**, with
**n_iter_no_change = 20**. The regularisation coefficient **alpha** is
**0.0003**, the initial learning rate is **0.00075**, and the maximum
iteration count is **10000**.

**The model is not a single neural network by default.** The backend
builds an ensemble of **n_models** networks, with the default set to
**10**. Each network is fitted on a bootstrap resample if more than one
model is requested. This is a bagging strategy. The effect is to reduce
variance, stabilise the prediction, and enable percentile outputs across
the ensemble.

**Why bagging is valuable here.** Overtopping datasets are noisy and
highly heterogeneous. A single MLP can be sensitive to the exact
training split and optimisation path. A bagged ensemble reduces
sensitivity to any one resample and creates a practical predictive
spread. That spread is not a full probabilistic uncertainty model, but
it is a much better operational output than a single point estimate with
no spread information.

**Pipeline used by each ensemble member:**

```text
SimpleImputer(strategy="median")
→ StandardScaler()
→ MLPRegressor(
    hidden_layer_sizes=(192, 96, 48, 24),
    activation="relu",
    solver="adam",
    alpha=0.0003,
    learning_rate_init=0.00075,
    early_stopping=True,
    validation_fraction=0.12,
    n_iter_no_change=20,
    max_iter=10000,
)
```

### 8.5 Holdout validation and full-data refit

**The current workflow uses a two-stage logic that deserves explicit
recognition.** First, it evaluates model performance on a **holdout
subset**. Second, after that evaluation, it **refits the final ensemble
on the full valid dataset** and saves that refitted model as the
production bundle. The GUI documentation correctly describes this
distinction and explicitly tells the user that holdout diagnostics
indicate validation performance, whereas the final saved model
corresponds to the refitted production model.

**Why this matters.** Many users assume that the model used for
diagnostics is the same fitted object later used in production. In the
current workflow, that is not exactly the case. The holdout stage
answers **"How well does the approach generalise on unseen data?"** The
full-data refit answers **"What model do I want to deploy after I have
judged the approach acceptable?"** Both are valid objectives, but they
should not be conflated.

**Holdout split details.** The default holdout fraction is **15%**. The
code attempts to create stratification bins from the log-transformed
target using quantile cuts, which is a sensible way to prevent the
validation subset from being dominated by only one part of the
overtopping range. If viable bins are not available, the split can still
proceed without stratification.

**Saved model metadata records this design choice.** The metrics include
a flag stating that the **saved model was refit on the full dataset**.
This is excellent practice because it prevents future confusion when a
user inspects the diagnostics file months later.

### 8.6 Diagnostics and reported metrics

**The diagnostics are not cosmetic.** They are the primary evidence that
training produced a reasonable model. The backend computes metrics in
three spaces: **$\log_{10}(s_q)$**, **sq**, and **q in l/s/m**. That is a
robust reporting choice, because it lets the user judge performance in
the transformed fitting space and in physical engineering units.

**Reported statistics include R2, MAE, Median AE, RMSE, and bias.** In
addition, the diagnostics store counts such as training rows, holdout
rows, full rows, number of bagged models, hidden layers, test size,
number of engineered features, and the fact that the saved model was
refit on the full dataset. The feature ranges are also stored in the
diagnostics JSON and later reused by the GUI for range checking.

**Prediction tables contain both custom non-dimensional and physical
outputs.** The printed prediction table includes **sq_mean**,
**sq_p05**, **sq_p50**, **sq_p95**, plus the corresponding **q**
estimates in **l/s/m**, together with a **range_warning** field. The
user should not ignore the range warning. It is one of the most
important outputs in the entire workflow.

### 8.7 Reconstruction of physical discharge from the model target

**The model is fitted in log-space, but the engineering result is
returned in physical discharge units.** After predicting **$\log_{10}(s_q)$**
for each ensemble member, the script exponentiates to recover **sq**,
then multiplies by **$\sqrt{g\,H_{m0,\mathrm{toe}}^{3}}$** to reconstruct **$q$ in m³/s/m**,
and finally multiplies by **1000** to express the result in **l/s/m**.

**This reconstruction is possible only if **$H_{m0,\mathrm{toe}}$** is finite and
positive.** The code explicitly checks that. If **$H_{m0,\mathrm{toe}}$** is missing
or non-positive, the script cannot recover physical discharge from the
dimensionless target, and that scenario is flagged as invalid in the
range warnings.

**Reconstruction logic used after prediction:**

$$
\begin{aligned}
\widehat{s_q} &= 10^{\widehat{\log_{10}(s_q)}} \\
\widehat{q}_{\mathrm{m^{3}/s/m}} &= \widehat{s_q} \, \sqrt{g \, H_{m0,\mathrm{toe}}^{3}} \\
\widehat{q}_{\mathrm{l/s/m}} &= 1000 \, \widehat{q}_{\mathrm{m^{3}/s/m}}
\end{aligned}
$$

### 8.8 Observed training-domain size in the project database

**The project pre-filtered `database.csv` file contains 9,334 rows and 
30 columns in total.** When the train.py validity filters are emulated 
exactly at the level of required overtopping rows, all rows remain for 
modelling after requiring valid **$q$** and valid positive 
**$H_{m0,\mathrm{toe}}$**. This number is important because the model's 
credibility depends not only on algorithm choice but also on the breadth 
of usable calibrated data.

**Observed ranges do not guarantee local density.** A variable can have
a very broad minimum-to-maximum range while still being very sparse in
some subregions. Therefore the range table later in Appendix E should be
interpreted as a **necessary but not sufficient** condition for safe
interpolation. A case lying within the min-max range may still be poorly
represented if the surrounding local data density is weak.

**Nevertheless, range checking is still valuable.** If a scenario lies
outside the observed range for one or more inputs, the operator knows
immediately that the model is extrapolating. The GUI exposes this check
clearly, which is a strong practical feature of the current workflow.

### 8.9 Engineering interpretation of the custom target space

**Because the model is trained on sq rather than directly on q, it
learns relative overtopping intensity after normalising by wave
height.** This tends to help transferability across conditions where the
absolute discharge changes strongly with $H_{m0,\mathrm{toe}}$. In effect, the model is asked first to learn how intense overtopping is relative to the
wave-height scaling, and only afterwards to reconstruct the dimensional
discharge.

**This design has two consequences.** First, the model is strongly
sensitive to the quality of the supplied **$H_{m0,\mathrm{toe}}$** value, because the
same variable appears in both the inputs and the target normalisation.
Second, the physical discharge reconstruction can change rapidly when
**$H_{m0,\mathrm{toe}}$** changes, even if the predicted sq changes only moderately.
Users should therefore treat **$H_{m0,\mathrm{toe}}$** as one of the most influential
and quality-critical inputs.

**This is also why the corrected target is easier to interpret.** The
script's internal **sq** is now directly comparable, in scaling terms,
with literature values based on **$q / \sqrt{g\,H^{3}}$**, provided the user
keeps track of the exact wave-height definition used in the numerator
scaling. In this workflow, the reference height is **$H_{m0,\mathrm{toe}}$**, so any
comparison with published values should respect that same toe-based
hydraulic definition.

> **Critical modelling note.** The current workflow uses **physically informed feature engineering** and **data-driven regression**, but it does **not** enforce hard physical monotonicity constraints. If a user performs wide parameter sweeps far from the training domain, some locally counterintuitive trends may still appear. That is a normal risk in flexible ML models and one more reason to inspect the range warning and the spread indicators carefully.

## 9. Detailed practical use of train.py

**The project `train.py` script is the authoritative computational backend.**
Even when the user works mainly through the GUI, the practical logic
still comes from train.py. Understanding it is therefore essential for
troubleshooting, automation, and defensible engineering use.

### 9.1 Files produced and consumed by the backend

Core files in the train.py workflow

- **database.csv**
  - **Role**: Training database
  - **When it is used**: Read during model training or auto-training.

- **model.joblib**
  - **Role**: Saved trained model bundle
  - **When it is used**: Loaded during prediction or replaced after retraining.

- **diagnostics.json**
  - **Role**: Metrics, metadata, feature ranges, plot paths
  - **When it is used**: Written after training and read by the GUI.

- **plots/...**
  - **Role**: Diagnostic plots
  - **When it is used**: Written after training to accompany diagnostics.

- **predictions.csv**
  - **Role**: Prediction output table
  - **When it is used**: Written after single or batch prediction.

- **input.txt**
  - **Role**: Default batch input file for the GUI and one CLI workflow
  - **When it is used**: Used for batch prediction when the operator chooses that format.

**model.joblib** is not just a fitted neural network weight file. It is
a complete saved **ModelBundle** containing the fitted pipelines, the
list of feature columns, the feature ranges, metrics, metadata, and
target definition information. Operationally, this is beneficial because
the prediction step carries its own context with it.

### 9.2 Preparing the environment

**For source use, create a dedicated Python environment.** The GUI help
text recommends a project-specific virtual environment. This is good
practice because machine-learning dependencies are sensitive to version
mismatches.

**A practical Windows sequence** is to create the environment, activate
it, upgrade pip, install the required packages, and then run the script
from the project directory. The GUI help text specifically lists
**numpy**, **pandas**, **matplotlib**, **scikit-learn**, **joblib**, and
**pyinstaller** among the operational packages for running and packaging
the workflow.

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
pip install pyinstaller numpy pandas matplotlib scikit-learn joblib
python train.py --help
```

### 9.3 Command structure

**The script has two top-level subcommands: `train` and `predict`.**
This is intentionally simple. The `train` mode creates or refreshes a
model bundle and diagnostics. The `predict` mode loads a saved model
if it exists, or can auto-train a new model if instructed with a
database path.

```bash
python train.py train [training arguments]
python train.py predict [prediction arguments]
```

### 9.4 Training a model

**Training is the first operation to run when no model exists or when
the database has changed.** The minimum required argument is the
database path. The model path and diagnostics path have defaults, but in
engineering work it is often preferable to state them explicitly for
traceability.

**The most standard training command** is the one documented in the
script header itself. This command reads **database.csv**, fits the
model, saves **model.joblib**, and writes **diagnostics.json**.

**Configurable training arguments.** The training subcommand accepts
`--sq-floor`, `--n-models`, `--max-iter`, and `--random-state`
in addition to the required database/model/diagnostics paths. These
arguments let the operator adjust the floor used before **$\log_{10}(s_q)$**,
the number of bagged MLP models, the optimisation budget, and the seed
controlling reproducibility.

```bash
python train.py train --database database.csv --model model.joblib ^
--diagnostics diagnostics.json
```

**What happens internally when this command runs.** The script loads the
database, filters valid rows, creates the custom target, engineers
additional features, performs a holdout split, trains the holdout-stage
ensemble, evaluates diagnostics, refits the final ensemble on the full
valid dataset, saves the bundle, writes diagnostics, and emits plots.

### 9.5 Understanding the training arguments

Training arguments in train.py

- **--database**
  - **Default**: Required
  - **Meaning**: Path to the training database CSV.

- **--model**
  - **Default**: model.joblib
  - **Meaning**: Path where the saved model bundle will be written.

- **--diagnostics**
  - **Default**: diagnostics.json
  - **Meaning**: Path where metrics, metadata, ranges, and plot paths will be written.

- **--sq-floor**
  - **Default**: 0.01
  - **Meaning**: Lower bound applied before $\log_{10}(s_q)$.

- **--n-models**
  - **Default**: 10
  - **Meaning**: Number of bagged MLP models in the ensemble.

- **--max-iter**
  - **Default**: 10000
  - **Meaning**: Maximum optimisation iterations for each MLP.

- **--random-state**
  - **Default**: 42
  - **Meaning**: Seed controlling reproducibility and bootstrap sampling.

### 9.6 Single-scenario prediction from the command line

**Single-scenario prediction is appropriate when an analyst needs one
manually specified case without using the GUI.** The script requires the
model path and the output path, plus the scenario inputs. The scenario
receives a name through `--name`, which is carried into the output
table.

**The script exposes all main physical inputs directly as
command-line arguments.** These include `--m`, `--beta`, `--h`,
`--hm0-toe`, `--tm-1-0-toe`, `--ht`, `--bt`, `--gf`,
`--cotad`, `--cotau`, `--berm-width`, `--hb`, `--rc`,
`--ac`, and `--gc`.

```bash
python train.py predict --model model.joblib --output ^
    predictions.csv --name case1 ^
    --m 30 --beta 0 --h 5 --hm0-toe 2.5 --tm-1-0-toe 8 ^
    --ht 5 --bt 0 --gf 0.55 --cotad 2 --cotau 2 ^
    --berm-width 0 --hb 0 --rc 3 --ac 3 --gc 3
```

**Output behaviour.** The script prints a compact table to the console
and then writes the full prediction table to the CSV specified by
`--output`. The printed compact table includes both the internal
**sq** outputs and the reconstructed **q** outputs, plus the range
warning field.

### 9.7 Batch prediction from input.txt or other scenario files

**Batch prediction is the preferred mode when the analyst has many
scenarios.** The script supports two mutually exclusive batch sources:
`--from-inp` for a pipe-delimited scenario file and `--from-csv` for
a CSV or semicolon-separated file. The GUI also supports broader
delimiter-based reading by calling backend helpers, but the CLI is
explicit about these two routes.

**The `--from-inp` mode** reads a pipe-delimited file with a header.
The parser is tolerant to several aliases. For example, it can derive
the single roughness factor **gf** by averaging separate **gammaf_d**
and **gammaf_u** values if those are present. This is useful when the
batch file comes from a format closer to earlier overtopping tools or
manually curated operator files.

**The `--from-csv` mode** reads a comma-separated or
semicolon-separated text file and then normalises common header aliases.
If no `Name` column is present, the script automatically generates
scenario names such as `scenario_1`, `scenario_2`, and so on.

- **python train.py predict --model model.joblib --from-csv scenarios.csv ^
  --output predictions.csv**

**A critical practical point.** The batch file must still supply the
physical variables expected by the backend. Header aliasing is helpful,
but it is not magical. The safest practice is to keep batch columns as
close as possible to the canonical variable names.

### 9.8 Auto-training when the model file is missing

**Auto-training is a convenience feature, not a substitute for
deliberate training review.** If the model file named by `--model`
does not exist, the function `ensure_bundle()` can trigger training
automatically, provided that a valid database path is supplied through
`--database`.

**This is useful for scripted workflows** in which the operator wants
`predict` to be robust to a missing model bundle. However, from a
controlled engineering perspective, it is usually better to run explicit
training first, review diagnostics, and only then move on to production
prediction.

```bash
python train.py predict --model model.joblib --database database.csv ^
--from-csv scenarios.csv --output predictions.csv
```

### 9.9 Meaning of the prediction columns

Columns emitted by the prediction table

- **Name**: Scenario identifier carried through the workflow.
- **sq_mean**: Mean predicted custom non-dimensional target sq.
- **sq_p05 / sq_p50 / sq_p95**:
  5th, 50th, and 95th percentile of predicted sq across the ensemble.
- **q(l/s/m)**: Mean predicted overtopping discharge in litres per second per metre.
- **q_p05(l/s/m) / q_p50(l/s/m) / q_p95(l/s/m)**:
  Ensemble percentile outputs in physical units.
- **range_warning**: Text description of any out-of-domain base input or missing $H_{m0,\mathrm{toe}}$ issue.

### 9.10 Recommended training and prediction workflow

**Step 1.** Train explicitly from the chosen database and save both
model and diagnostics.

**Step 2.** Open diagnostics.json and the plots folder; verify that the
holdout metrics are reasonable.

**Step 3.** Preserve the trained model as a versioned artefact if the
project requires traceability.

**Step 4.** Run a small number of known cases first and inspect both the
q values and the range warnings.

**Step 5.** For batch work, check the header names before submitting
hundreds of scenarios.

**Step 6.** After every large batch, scan the warnings column before
trusting the results table.

## 10. Detailed practical use of gui.py and gui.exe

**The project `gui.py` script is a Tkinter desktop interface built specifically
around the `train.py` backend.** Operationally, the GUI is the
easiest way for most engineers to interact with the model because it
exposes the prediction, training, batch, diagnostics, and range-check
workflows in one desktop window.

**The same practical guidance applies to gui.exe if gui.py has been
compiled to a Windows executable from the same revision.** The
executable is simply the packaged delivery form of the same logic. What
matters is that the packaged executable and the backend code remain
aligned.

### 10.1 What the GUI is designed to do

GUI functional objectives

- **Single-case prediction**: Enter one scenario manually and obtain mean q plus ensemble spread indicators.
- **Batch prediction**: Load a scenario file, predict all cases, preview results, and save a CSV.
- **Model training / refresh**: Train a new model from database.csv and regenerate diagnostics.
- **Calibration-domain review**: Compare current inputs against min-max ranges stored in the trained model bundle.
- **Persistent working state**: Restore prior inputs, paths, selected tab, and window geometry from defaults.json.

### 10.2 Startup behaviour

**Automatic startup loading is deliberate.** The GUI help text states
that when the GUI starts, it immediately tries to load the model file
shown in the `Model file` field. If that file exists and is readable,
the model is loaded before the user presses any prediction button. This
is a convenience feature that turns the desktop app into a
near-immediate prediction tool.

**If the model does not exist, the GUI still opens in a usable state.**
The interface remains ready so the user can either train a new model,
point the GUI to another existing bundle, or proceed later after
preparing the backend files.

**Persistent state means the GUI reopens close to the last session.**
The GUI writes a file called `defaults.json` and later reads it on
startup. Stored values include single-case inputs, model/database/output
paths, batch input and output paths, selected notebook tab, and last
window geometry. This is extremely useful for iterative engineering
work, but it also means the user must remain aware of what prior
settings were restored.

### 10.3 File fields used by the GUI

Principal file path controls in the GUI

- **Model file**: The saved `model.joblib` bundle used for prediction.
- **Database CSV**: The source database used when training or refreshing the model.
- **Single CSV output**: Destination CSV for the latest single-case prediction.
- **Diagnostics JSON**: Summary file written by the backend after training and later displayed by the GUI.
- **Batch input file**: Scenario file used in batch prediction; defaults to `input.txt` at startup.
- **Batch output CSV**: Destination CSV for the latest batch prediction results.

### 10.4 Single Prediction tab: complete operating procedure

**The Single Prediction tab is the primary operator screen.** The
left-hand side is used for entering or reviewing the current case and
the relevant file paths. The right-hand side is used for results
interpretation: prediction summary, diagnostics summary, warning panel,
and a review table of the current case.

**Input grouping is physically meaningful.** The GUI groups inputs as
**Case / Wave**, **Toe / Slope**, and **Berm / Crest**. That grouping is
not merely cosmetic. It helps the operator review the completeness of
the hydraulic forcing and geometric definition in an engineering
sequence.

**Recommended operator sequence.** First confirm that the correct model
is loaded. Second enter or revise all physical inputs. Third confirm the
single output CSV path if you want a saved result. Fourth press
**Predict q**. Fifth read the summary cards carefully. Sixth inspect
warnings and the review table. Seventh save the result if needed.

**Result interpretation on this tab.** The GUI documents the fields
clearly: **Average q [l/s/m]** is the main predicted discharge,
**P05/P50/P95 [l/s/m]** are the spread indicators, **Average q
[m³/s/m]** is the same discharge in SI units, and the warning panel
indicates whether the case lies outside the trained domain.

**Professional reading of P05, P50, and P95.** These should not be
treated as guaranteed lower, median, and upper physical truths. They are
percentile summaries of the custom ensemble output. Their value lies in
telling the user whether the current case is producing a tight or broad
ensemble response. Broad spread means the tool itself is less stable for
that case, which should lower confidence in blind use of the mean
estimate.

**Single-case step 1.** Review or enter `Name`, $H_{m0,\mathrm{toe}}$, $T_{m-1,0,\mathrm{toe}}$, `m`, `h`, and $\beta$.

**Single-case step 2.** Review or enter `ht`, `Bt`, `gf`,
`cotad`, and `cotau`.

**Single-case step 3.** Review or enter `B`, `hb`, `Rc`, `Ac`,
and `Gc`.

**Single-case step 4.** Confirm `model.joblib`, `database.csv`,
`diagnostics.json`, and the single-result output CSV path.

**Single-case step 5.** Press **Predict q**.

**Single-case step 6.** Read the summary cards and the SI conversion
field.

**Single-case step 7.** Inspect the **range warning** and the row-by-row
input review before accepting the result.

**Single-case step 8.** Press **Save single result** if a CSV output is
required for record keeping.

### 10.5 Batch Prediction tab: complete operating procedure

**The Batch Prediction tab is intended for many scenarios at once.** The
GUI documentation states that accepted formats include pipe-separated
text, tab-separated text/TSV, CSV, semicolon-separated text, and files
that can be parsed through the backend helpers.

**The default startup batch file is `input.txt`.** This is a helpful
convention because it gives the operator a known initial path every time
the GUI opens. If the current project uses a different batch file, the
operator should consciously overwrite or browse away from the default
path rather than assuming the field is already correct.

**The batch result preview is part of the quality-control workflow.**
After the run, the table preview shows each scenario name, mean q, P05,
P50, P95, and any range warning. This preview is not decorative. It is
the operator's first chance to detect failed input interpretation,
obviously incorrect magnitudes, or widespread extrapolation warnings
before exporting the final CSV.

**Batch step 1.** Select the batch input file.

**Batch step 2.** Confirm or edit the batch output CSV path.

**Batch step 3.** Press **Predict from batch file**.

**Batch step 4.** Inspect the preview table carefully, especially the
warning column.

**Batch step 5.** Use **Save batch result** to export the latest batch
output.

**Batch step 6.** Use **Use default input.txt** if you need to restore
the standard startup file path.

### 10.6 Training and diagnostics from the GUI

**The GUI is not prediction-only.** It can call the backend training
routine directly. This is important for non-programmer operators who
still need to refresh the model after the database changes.

**The documented GUI workflow for training** is straightforward: select
the database, confirm model and diagnostics paths, press **Train /
refresh model**, wait for the backend to complete, then allow the newly
trained model to load into the GUI.

**The diagnostics summary uses the backend's two-stage logic.** The GUI
explicitly distinguishes **valid rows**, **n_holdout**, **n_full**,
**R2(log10 sq)**, and **R2(q l/s/m)**. Users must understand that
`n_holdout` refers to the validation subset and `n_full` refers to
the final refit dataset used for the saved production model.

**Best practice after retraining** is not merely to trust that training
completed. The operator should review both the model summary and
diagnostics summary and, if possible, the plots referenced by
diagnostics.json.

### 10.7 Parameter Ranges tab

**This tab is one of the strongest safety features in the interface.**
It compares the currently entered numeric inputs against the observed
training-domain minimum and maximum stored in the loaded model bundle.

**Interpretation of statuses.** `inside` means the current value lies
within the observed database range. `outside` means the model is
extrapolating in that variable. `invalid` means the entry is not
numerically valid. `range unavailable` means the saved model does not
provide a usable range for that field. `model not loaded` means no
bundle is currently available for comparison.

**Why range review should be done before prediction rather than after.**
If a case is obviously outside the trained domain, the user may choose
to adjust the scenario, verify the data entry, or reframe the modelling
strategy before obtaining a potentially misleading estimate.

### 10.8 defaults.json and state persistence

**defaults.json is operationally convenient but procedurally important.**
Because it restores prior inputs and file paths, it helps long-running
engineering studies. But in regulated or high-consequence work it also
means the operator should verify that the restored paths still
correspond to the intended current project files.

**Reset logic.** The GUI documentation states that if the user wants a
full reset, the GUI should be closed and `defaults.json` deleted. The
next launch will then restore built-in defaults instead of the last
session state.

### 10.9 Running gui.py from source and using gui.exe

**Source execution** is appropriate when the environment is managed by a
technical user or when the backend may still be evolving. **Executable
use** is appropriate when the workflow is being delivered to an end user
who should not need to interact with Python directly.

**In either case, the engineering workflow is the same.** The operator
still loads or trains a model, enters a case or selects a batch file,
inspects warnings, interprets percentile outputs, and saves results.

```bash
python gui.py
```

## 11. Interpretation of outputs, diagnostics, percentiles, and warnings

**A prediction is only as useful as its interpretation.** The current
workflow returns numbers that look precise, often with several decimal
places, but engineering interpretation must remain qualitative and
conditional.

**Mean q is the operational centre of gravity.** For most users, the
primary result is the mean overtopping discharge in **l/s/m** or
**m³/s/m**. This is the quantity used for preliminary comparison against
tolerability criteria, conceptual design checks, and ranking of
alternatives.

**P05, P50, and P95 should be read as spread descriptors of the custom
ensemble.** They are useful because they immediately tell the operator
whether the ensemble members agree closely or not. A narrow spread
suggests stable interpolation. A wide spread suggests sensitivity, weak
support, or some internal disagreement in the ensemble response. The
wider the spread, the more conservative the operator should be in using
the mean estimate as a stand-alone number.

**The diagnostics metrics should be read in layers.** `R2(log10 sq)`
indicates how well the model is fitting in the transformed target space
it actually optimises. `R2(q l/s/m)` indicates how well it represents
overtopping in the physical quantity that engineers use. MAE and RMSE
indicate typical and more outlier-sensitive error magnitude, while bias
indicates whether the model tends to overpredict or underpredict on
average.

**Holdout metrics are more important than refit metrics for
credibility.** The refit metrics tell the user how well the final model
aligns with the full dataset on which it was refit. The holdout metrics
tell the user whether the modelling strategy can generalise to unseen
samples. In other words, when judging whether the workflow is
trustworthy, the operator should weight holdout metrics more heavily
than full-data refit plots.

**Range warnings are not optional metadata.** If the warning says that
one or more variables are outside the observed range, the prediction
should be described explicitly as an extrapolation. This is still
sometimes useful in screening studies, but the operator should avoid
presenting such a result as if it had the same credibility as an
in-range interpolation.

**The absence of a warning is not proof of local data density.** A case
can be inside the min-max domain and yet still lie in a sparse corner of
the database. Therefore an apparently clean range status is a necessary
but incomplete credibility test. Use it as the first gate, not the only
gate.

How to read key outputs professionally

- **Mean q**
  - **Naive reading**: The exact overtopping discharge
  - **Correct engineering reading**: The central estimate produced by the trained ensemble under the supplied inputs.

- **P05 / P95**
  - **Naive reading**: Guaranteed lower and upper physical bounds
  - **Correct engineering reading**: Low and high ensemble percentile descriptors; useful for spread, not absolute truth.

- **R2**
  - **Naive reading**: The model is accurate everywhere
  - **Correct engineering reading**: Global fit indicator that must be combined with range review and engineering judgement.

- **No range warning**
  - **Naive reading**: Safe result
  - **Correct engineering reading**: Necessary first check passed; local sparsity may still exist.

- **Range warning**
  - **Naive reading**: The result is useless
  - **Correct engineering reading**: Extrapolation flag; the result may still be informative, but confidence should be downgraded.

## 12. Engineering limitations, applicability, and best practice

**Every overtopping predictor is conditional on its database and
assumptions.** This statement applies to empirical equations, classic
ANN tools, XGB tools, and the current custom MLP ensemble alike. It is
particularly important for overtopping because the response can change
by orders of magnitude across realistic design space and because small
changes in geometry may strongly affect run-up and discharge.

**The current workflow is best used for conceptual design, comparative
screening, and rapid iterative assessment.** It is well suited to tasks
such as: ranking crest freeboard alternatives; comparing the influence
of roughness, berm width, or obliquity; screening many scenarios from a
forecast or design matrix; and producing preliminary estimates before
committing to more expensive analysis.

**The workflow is less appropriate as a sole basis for final design
decisions** when the case is highly unusual, when the structure has
features not well represented in the database, when the project is
high-consequence, or when the case sits near extreme overtopping levels
where database coverage is usually thinner.

**Local engineering judgement remains essential.** If a result appears
physically implausible relative to known overtopping behaviour, do not
hide behind the fact that a neural network produced it. Re-check the
inputs. Compare against a representative empirical estimate. Inspect the
range status. Review the ensemble spread. Then decide whether the
scenario should be treated as a suspect extrapolation or as a prompt for
deeper analysis.

**Database imbalance matters.** Recent machine-learning studies
emphasise that larger overtopping discharges are often less well
represented than smaller ones. That means that the model can appear
globally competent while still being weaker in the extreme upper tail.
Engineers should keep that asymmetry in mind whenever the case of
interest is a high-consequence near-failure or strong-overtopping case.

**The safest operational rule is hybrid verification.** For in-range
routine cases, a trained ML estimate may be entirely adequate for
preliminary decisions. For important design choices, compare the ML
estimate against at least one classical formula of the relevant
structural family and ask whether the order of magnitude and trend are
mutually consistent. For out-of-range or unusual cases, escalate to more
rigorous methods.

**Best-practice rule 1.** Use measured or well-transformed toe
conditions, not offshore values passed through without thought.

**Best-practice rule 2.** Check all units before training or prediction;
overtopping tools are extremely sensitive to inconsistent input units.

**Best-practice rule 3.** Treat $H_{m0,\mathrm{toe}}$ and $R_c$ as
quality-critical variables because they dominate scaling and resistance.

**Best-practice rule 4.** Review the parameter ranges before running
production batches.

**Best-practice rule 5.** Do not report the mean alone when the spread
is wide; include the percentile context and note the warning status.

**Best-practice rule 6.** Preserve model version, diagnostics, and
database version together when traceability matters.

**Best-practice rule 7.** Re-train only when the database has materially
changed or when a documented model refresh is intended.

**Best-practice rule 8.** Keep gui.py and train.py from the same
revision so the interface and backend remain schema-compatible.

> **Final engineering stance.** A good overtopping ML workflow is not a replacement for engineering. It is a **force multiplier for engineering judgement**. It becomes dangerous only when the operator stops checking whether the case lies inside the learned domain and whether the output still respects first-order physical expectations.

## 13. Troubleshooting and operator checklists

### 13.1 Common technical and engineering problems

Troubleshooting matrix

- **Prediction cannot reconstruct q**
  - **Likely cause**: $H_{m0,\mathrm{toe}}$ missing, invalid, or \<= 0
  - **Recommended action**: Correct the wave-height input; the script cannot rescale sq to physical discharge without a valid $H_{m0,\mathrm{toe}}$.

- **Many scenarios show range warnings**
  - **Likely cause**: Batch file contains out-of-domain cases or incorrect units
  - **Recommended action**: Check units, headers, and whether the scenario family is genuinely outside the trained database.

- **Model file not found**
  - **Likely cause**: No trained bundle at the specified path
  - **Recommended action**: Run explicit training or use auto-training with a valid database path.

- **Diagnostics look good but one case looks implausible**
  - **Likely cause**: Case may be a sparse in-range point or a bad input combination
  - **Recommended action**: Cross-check inputs, review spread, compare with a formula-based estimate, and inspect range status carefully.

- **Batch results all look identical or nearly identical**
  - **Likely cause**: Header mapping failure or missing input columns leading to NaNs/imputation dominance
  - **Recommended action**: Re-check the batch file headers and confirm all required variables are populated.

- **GUI reopens with wrong paths**
  - **Likely cause**: defaults.json restored a previous session state
  - **Recommended action**: Correct the fields manually or delete defaults.json for a clean reset.

- **Training takes long**
  - **Likely cause**: Large ensemble, many rows, high max_iter
  - **Recommended action**: Wait for completion, or reduce n_models / max_iter deliberately and document the change.

- **Unexpected difference between holdout view and deployed behaviour**
  - **Likely cause**: Saved model is refit on the full valid dataset after validation
  - **Recommended action**: Interpret this as intended workflow design; use holdout metrics for credibility and full refit for deployment.

### 13.2 Pre-run checklist for train.py

\- Confirm that `database.csv` is the intended current training
database.

\- Confirm that the database contains the required 15 feature columns
plus `q`.

\- Confirm that units are consistent with the rest of the project
workflow.

\- Choose explicit output names for `model.joblib` and
`diagnostics.json` if version control matters.

\- Decide whether defaults (`sq_floor`, `n_models`, `max_iter`,
`random_state`) are appropriate for the current task.

### 13.3 Pre-run checklist for the GUI

\- Confirm that the restored defaults correspond to the current project,
not a previous session.

\- Confirm the loaded model version before reading any prediction as
current.

\- Check the `Database CSV` path before pressing `Train / refresh
model`.

\- Check the batch file path before pressing `Predict from batch
file`.

\- Review the `Parameter Ranges` tab before accepting a sensitive
result.

### 13.4 Post-run checklist for any prediction

\- Read the mean q in both l/s/m and m³/s/m if the interface provides
both.

\- Inspect P05, P50, and P95 to judge spread.

\- Read the range warning text fully.

\- Check whether the result magnitude is physically plausible against
engineering intuition.

\- Save the result table or screenshot the critical summary if
traceability is required.

## Appendix A. Formula compendium

**This appendix collects the main formulas that are most useful for
interpreting the current workflow.** The formulas below are presented
as a compact engineering reference, not as a substitute for reading the
full EurOtop chapters.

Formula sheet

- **Spectral significant wave height**: $H_{m0} = 4 \, \sqrt{m_0}$  
  **Why it matters**: Preferred overtopping wave-height variable at the toe.

- **Spectral period**: $T_{m-1,0} = \frac{m_{-1}}{m_0}$  
  **Why it matters**: Preferred overtopping period variable.

- **Toe wavelength**: $L_{m-1,0} = \frac{g \, T_{m-1,0}^{2}}{2 \pi}$  
  **Why it matters**: Used for wavelength-based normalisation.

- **Wave steepness**: $s = \frac{H_{m0}}{L_0}$  
  **Why it matters**: Compact wave-shape descriptor.

- **Breaker parameter**: $\xi_{m-1,0} = \frac{\tan(\alpha)}{\sqrt{H_{m0} / L_{m-1,0}}}$  
  **Why it matters**: Regime indicator for breaking / surging behaviour.

- **Generic overtopping form**: $\frac{q}{\sqrt{g \, H_{m0}^{3}}} = a \, \exp\!\left[-\left(b \, \frac{R_c}{H_{m0}}\right)^{c}\right]$  
  **Why it matters**: Weibull-shaped conceptual overtopping relation.

- **Smooth-slope breaking-wave context**: $\frac{q}{\sqrt{g \, H_{m0}^{3}}} = 0.023 \, \frac{\sqrt{\tan(\alpha)}}{\gamma_b \, \xi} \, \exp\!\left[-\left(\frac{2.7 \, R_c}{\xi \, H_{m0} \, \gamma_b \, \gamma_f \, \gamma_{\beta} \, \gamma_v}\right)^{1.3}\right]$  
  **Why it matters**: Mean-value overtopping context for breaking waves.

- **Smooth-slope non-breaking context**: $\frac{q}{\sqrt{g \, H_{m0}^{3}}} = 0.09 \, \exp\!\left[-\left(\frac{1.5 \, R_c}{H_{m0} \, \gamma_f \, \gamma_{\beta} \, \gamma_*}\right)^{1.3}\right]$  
  **Why it matters**: Upper-limit / non-breaking context.

- **Custom `train.py` target**: $s_q = \frac{q}{\sqrt{g \, H_{m0,\mathrm{toe}}^{3}}}$  
  **Why it matters**: Internal target used by the current model.

- **Custom `train.py` training target**: $y = \log_{10}(\max(s_q, s_{q,\mathrm{floor}}))$  
  **Why it matters**: Regression target used by the MLP ensemble.

- **Custom physical reconstruction**: $q = s_q \, \sqrt{g \, H_{m0,\mathrm{toe}}^{3}}$  
  **Why it matters**: Recover q in m³/s/m after prediction.

- **Litres-per-second conversion**: $q_{\mathrm{l/s/m}} = 1000 \, q$  
  **Why it matters**: Displayed engineering unit in the GUI summary.

## Appendix B. Glossary of symbols and variables

Glossary

- **q**: Mean overtopping discharge.
- **$H_{m0,\mathrm{toe}}$**: Spectral significant wave height at the toe of the structure.
- **$T_{m-1,0,\mathrm{toe}}$**: Spectral mean period at the toe of the structure.
- **m**: Foreshore slope cotangent in the CLI help.
- **$\beta$ / $b$**: Wave obliquity angle in degrees.
- **h**: Water depth at the toe/front of the structure.
- **ht**: Toe water depth.
- **Bt**: Toe width.
- **gf**: Roughness/permeability factor.
- **cotad**: Lower slope cotangent.
- **cotau**: Upper slope cotangent.
- **B**: Berm width.
- **hb**: Berm water depth or berm-level descriptor from the dataset convention.
- **Rc**: Crest freeboard.
- **Ac**: Armour crest freeboard.
- **Gc**: Crest width.
- **sq**: Custom non-dimensional target in the script: $q / \sqrt{g\,H_{m0,\mathrm{toe}}^{3}}$.
- **P05 / P50 / P95**: 5th, 50th, and 95th ensemble percentile outputs.
- **n_holdout**: Number of rows used for holdout validation diagnostics.
- **n_full**: Number of rows used in the final full-data refit.

## Appendix C. train.py CLI quick-reference

Command patterns

```bash
python train.py train \
  --database database.csv \
  --model model.joblib \
  --diagnostics diagnostics.json

python train.py predict \
  --model model.joblib \
  --output predictions.csv \
  --name case1 ...

python train.py predict \
  --model model.joblib \
  --from-inp input.txt \
  --output predictions.csv

python train.py predict \
  --model model.joblib \
  --from-csv scenarios.csv \
  --output predictions.csv

python train.py predict \
  --model model.joblib \
  --database database.csv \
  --from-csv scenarios.csv \
  --output predictions.csv
```

Selected CLI arguments

- **--m**
  - **Mode**: Single-case predict
  - **Meaning**: Foreshore slope cotangent

- **--beta**
  - **Mode**: Single-case predict
  - **Meaning**: Wave obliquity in degrees

- **--h**
  - **Mode**: Single-case predict
  - **Meaning**: Water depth at toe/front [m]

- **--hm0-toe**
  - **Mode**: Single-case predict
  - **Meaning**: $H_{m0,\mathrm{toe}}$ at toe [m]

- **--tm-1-0-toe**
  - **Mode**: Single-case predict
  - **Meaning**: $T_{m-1,0,\mathrm{toe}}$ at toe [s]

- **--ht**
  - **Mode**: Single-case predict
  - **Meaning**: Toe water depth [m]

- **--bt**
  - **Mode**: Single-case predict
  - **Meaning**: Toe width [m]

- **--gf**
  - **Mode**: Single-case predict
  - **Meaning**: Roughness/permeability factor [-]

- **--cotad**
  - **Mode**: Single-case predict
  - **Meaning**: Lower slope cotangent [-]

- **--cotau**
  - **Mode**: Single-case predict
  - **Meaning**: Upper slope cotangent [-]

- **--berm-width**
  - **Mode**: Single-case predict
  - **Meaning**: Berm width B [m]

- **--hb**
  - **Mode**: Single-case predict
  - **Meaning**: Berm water depth hb [m]

- **--rc**
  - **Mode**: Single-case predict
  - **Meaning**: Crest freeboard [m]

- **--ac**
  - **Mode**: Single-case predict
  - **Meaning**: Armour crest freeboard [m]

- **--gc**
  - **Mode**: Single-case predict
  - **Meaning**: Crest width [m]

- **--from-inp**
  - **Mode**: Batch predict
  - **Meaning**: Pipe-separated scenario file

- **--from-csv**
  - **Mode**: Batch predict
  - **Meaning**: CSV or semicolon-separated scenario file

- **--sq-floor**
  - **Mode**: Train / auto-train
  - **Meaning**: Lower bound before $\log_{10}(s_q)$

- **--n-models**
  - **Mode**: Train / auto-train
  - **Meaning**: Number of bagged MLP models

- **--max-iter**
  - **Mode**: Train / auto-train
  - **Meaning**: Maximum MLP iterations

- **--random-state**
  - **Mode**: Train / auto-train
  - **Meaning**: Random seed

## Appendix D. GUI quick-reference

Main GUI buttons

- **Predict q**: Runs a single-case prediction using the current model and current inputs.
- **Train / refresh model**: Fits a new model from the selected database and regenerates diagnostics.
- **Save single result**: Exports the latest single prediction as CSV.
- **Predict from batch file**: Runs all scenarios from the selected batch file.
- **Save batch result**: Exports the latest batch result table as CSV.
- **Use default input.txt**: Restores the standard startup batch file path.
- **Reset defaults**: Restores built-in default physical inputs.
- **Open ranges tab / Open batch tab**:
  Quick navigation shortcuts to review domain status or batch workflow.

Main GUI tabs

- **Single Prediction**: Manual case entry and immediate prediction summary.
- **Batch Prediction**: Many scenarios at once, with preview and CSV export.
- **Parameter Ranges**: Per-variable inside/outside review against the trained domain.
- **Log**: Traceability view of what was loaded, trained, predicted, or saved.

## Appendix E. Observed training-domain ranges from the project database

**The following range summary reports the observed min, max, and median
values after emulating the backend validity filters used for overtopping
training rows.** It is based on approximately **10,392 valid rows** from
the project database. Use this appendix as a **range reference**, not
as proof of local data density.

Observed ranges for base features

- **m**
  - **Min**: 6
  - **Max**: 1000
  - **Median**: 100

- **b**
  - **Min**: 0
  - **Max**: 80
  - **Median**: 0

- **h**
  - **Min**: 0.016
  - **Max**: 9.32
  - **Median**: 0.4

- **$H_{m0,\mathrm{toe}}$**
  - **Min**: 0.003
  - **Max**: 3.765
  - **Median**: 0.112

- **$T_{m-1,0,\mathrm{toe}}$**
  - **Min**: 0.495
  - **Max**: 10.64
  - **Median**: 1.532

- **ht**
  - **Min**: 0
  - **Max**: 7.78
  - **Median**: 0.3

- **Bt**
  - **Min**: 0
  - **Max**: 10
  - **Median**: 0

- **gf**
  - **Min**: 0.33
  - **Max**: 1
  - **Median**: 0.7

- **cotad**
  - **Min**: 0
  - **Max**: 7.07
  - **Median**: 2

- **cotau**
  - **Min**: -5
  - **Max**: 9.71
  - **Median**: 1.72

- **B**
  - **Min**: 0
  - **Max**: 8
  - **Median**: 0

- **hb**
  - **Min**: -0.208333
  - **Max**: 1.175
  - **Median**: 0

- **Rc**
  - **Min**: 0
  - **Max**: 8.345
  - **Median**: 0.148

- **Ac**
  - **Min**: 0
  - **Max**: 7.87
  - **Median**: 0.132

- **Gc**
  - **Min**: 0
  - **Max**: 5.6
  - **Median**: 0.032

## Appendix F. Parameter-by-parameter operating notes

**m.** Use **m** exactly as defined in the project data convention. In
the CLI help it is described as the **foreshore slope
cotangent**. Because the script uses **m** as a base feature, do not
silently swap it with another slope descriptor from external tools
without verifying the mapping.

**$\beta$.** Enter the wave obliquity in **degrees**. The backend will
derive **`beta_abs`**, **$\cos(\beta)$**, and **$|\sin(\beta)|$** from this
field. A sign mistake in obliquity can change directional interaction
terms even if the magnitude looks reasonable.

**h.** This is a **local depth input**, not a generic offshore
descriptor. If the project uses transformed toe conditions, keep the
depth consistent with those toe conditions.

**$H_{m0,\mathrm{toe}}$.** Treat this as one of the **most influential inputs** in the
entire workflow. It appears both as a model input and in the target
scaling. Errors in **$H_{m0,\mathrm{toe}}$** can therefore distort the prediction
twice: once in feature space and once in physical reconstruction.

**$T_{m-1,0,\mathrm{toe}}$.** This variable controls the wavelength scaling used to
form several engineered features. A period entered in the wrong units or
taken from the wrong spectral definition will propagate through many
derived ratios.

**ht.** Use the toe depth consistently with the schematisation adopted
in the project. Small changes in toe definition can be important for
regime classification and geometric representation.

**Bt.** Toe width is particularly important in composite geometries. If
the toe is effectively absent, represent that intentionally rather than
leaving an unknown value.

**gf.** Do not treat **gf** as a cosmetic coefficient. It is a
roughness/permeability control that can materially affect overtopping.
Use a value consistent with the structure type and the project's adopted
convention.

**cotad.** This lower slope cotangent is used to reconstruct a tangent
and build **xi_m10_lower**. Avoid entering a slope angle where a
cotangent is expected.

**cotau.** This upper slope cotangent is used similarly for
**xi_m10_upper**. Negative or uncommon values deserve special care and
should be checked against the geometry convention in the dataset.

**B.** Berm width should be specified consistently with the project's
schematisation. If no berm is present, use the correct no-berm
representation rather than inventing a nominal small value.

**hb.** The berm vertical descriptor should be entered according to the
dataset's meaning. Because conventions vary between tools, this is a
field that deserves explicit review whenever data are exchanged between
software packages.

**Rc.** Crest freeboard is a first-order resistance variable. If **Rc**
is wrong, the overtopping estimate can become grossly misleading even if
all other inputs are reasonable.

**Ac.** Armour crest freeboard provides additional crest-elevation
information. It should not be left inconsistent with **Rc**.

**Gc.** Crest width affects the horizontal extent of the crest region.
Its influence is often secondary to freeboard but still meaningful,
especially in composite geometries and post-run-up passage behaviour.

## Appendix G. Example operator workflows

**Workflow 1 - First-time setup.**

1\. Place `train.py`, `gui.py`, `database.csv`, and `input.txt`
in the project directory.

2\. Create and activate the Python environment.

3\. Run a first explicit training command to create `model.joblib` and
`diagnostics.json`.

4\. Open the GUI only after the first model exists, so the interface can
load immediately.

5\. Check the ranges and run one known benchmark scenario before
production use.

**Workflow 2 - Daily GUI use for one scenario.**

1\. Open `gui.py` or `gui.exe` and let it load the last model
automatically.

2\. Verify that the restored paths in `defaults.json` still point to
the correct project files.

3\. Edit the current case in the Single Prediction tab.

4\. Review the Parameter Ranges tab.

5\. Run `Predict q`, inspect mean q and percentile spread, and save
the result if needed.

**Workflow 3 - Batch study with many scenarios.**

1\. Prepare a batch file with the required headers and units.

2\. Open the Batch Prediction tab and confirm the current batch file
path.

3\. Run the batch prediction and inspect the preview table before
export.

4\. Scan for widespread warnings or suspiciously identical outputs.

5\. Export the batch CSV only after the preview looks credible.

**Workflow 4 - Model refresh after database change.**

1\. Replace or update `database.csv` deliberately and archive the
previous version if traceability matters.

2\. Run `Train / refresh model` from the GUI or `train.py train ...`
from the command line.

3\. Review `diagnostics.json`, holdout metrics, and plots before
accepting the new model.

4\. Record the updated model file name or folder if the project requires
version control.

## Appendix H. Detailed interpretation of diagnostics.json

**diagnostics.json is the machine-readable summary of model quality.**
It stores four broad classes of information: **metrics**, **metadata**,
**feature_ranges**, and **plot_paths**. A disciplined workflow should
preserve this file together with the corresponding model bundle, because
the model without diagnostics is difficult to audit later.

**metrics** contains the quantitative fit statistics. In the current
backend, these include row counts, R2, MAE, Median AE, RMSE, bias, the
target floor, number of models, hidden-layer structure, test size,
number of engineered features, and MLP hyperparameters. When a model is
refreshed, this section is the fastest way to determine whether the new
version is comparable to or better than the previous version.

**metadata** explains what the model actually is. This includes the
database path used, the model type string, the complete list of feature
columns, the base feature columns, the target column name, the target
definition, and the reported output units. For long-term
maintainability, the metadata section is almost as important as the raw
scores.

**feature_ranges** is operationally important because it is what the GUI
uses to warn about extrapolation. If a user loads a model bundle from an
incorrect project or from an older revision, the range logic in the GUI
will also change. That is one more reason to keep project artefacts
organised.

**plot_paths** points to the diagnostic plot files generated under the
`plots` directory. These plots provide visual context for holdout and
full-data performance. A metrics-only review is acceptable for quick
checks, but a proper model review should also look at the plots.

Most important diagnostics.json fields

- **metrics.train_rows / test_rows / holdout_rows / full_rows**:
  How many rows were used in each stage of the workflow.
- **metrics.r2_log10_sq**: Fit quality in the transformed training space.
- **metrics.r2_q_lpsm**: Fit quality in physical discharge units.
- **metrics.bias_q_lpsm**: Average physical overprediction or underprediction tendency.
- **metrics.n_models**: Number of bagged MLP models used in the ensemble.
- **metrics.hidden_layers**: Network architecture used in the current model version.
- **metadata.model_type**: One-line description of the modelling strategy.
- **metadata.target_definition**: Explicit statement of the internal target; must be checked carefully.
- **feature_ranges**: Min-max training domain used later by the GUI range checker.
- **plot_paths**: Locations of the generated plot files.

## Appendix I. Command and usage recipe library

**Recipe 1 - Rebuild the model from scratch.**

```bash
python train.py train ^
    --database database.csv ^
    --model model.joblib ^
    --diagnostics diagnostics.json ^
    --n-models 10 ^
    --max-iter 10000 ^
    --random-state 42
```

**Recipe 2 - Predict one manual case after training.**

```bash
python train.py predict ^
    --model model.joblib ^
    --output predictions.csv ^
    --name test_case ^
    --m 30 --beta 0 --h 5 --hm0-toe 2.5 --tm-1-0-toe 8 ^
    --ht 5 --bt 0 --gf 0.55 --cotad 2 --cotau 2 ^
    --berm-width 0 --hb 0 --rc 3 --ac 3 --gc 3
```

**Recipe 3 - Predict a batch from the default-style input file.**

```bash
python train.py predict ^
    --model model.joblib ^
    --from-inp input.txt ^
    --output predictions.csv
```

**Recipe 4 - Predict a CSV batch and auto-train if needed.**

```bash
python train.py predict ^
    --model model.joblib ^
    --database database.csv ^
    --from-csv scenarios.csv ^
    --output predictions.csv
```

**Recipe 5 - Run the GUI from source.**

```bash
python gui.py
```
