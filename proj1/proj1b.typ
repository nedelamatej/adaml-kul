// [ADAML] Advanced Data Analysis and Machine Learning
//         Project 1b
//
// Lappeenranta-Lahti University of Technology
// School of Engineering Science
//
// Name: proj1b.typ
// Aut.: Moriom Akter
//       Matej Nedela
//       Monowarul Sabbir
// Date: 27/09/2026
// Ver.: 1.0

#set text(size: 9.5pt)
#set page(margin: 1.25cm)

#set document(
  title: "Advanced Data Analysis and Machine Learning (Project 1b)",
  author: ("Moriom Akter", "Matěj Neděla", "Monowarul Sabbir"),
  date: datetime(year: 2026, month: 9, day: 27),
)

#context align(center)[
  #block[#text(weight: 700, 1.6em, document.title)]
  #v(-1em * 2/3)
  #block[#text(weight: 500, 1.0em, fill: black.lighten(33%))[--- *NASA Turbofan Jet Engine* (level A): _Forecast when the engine is due to maintenance_ ---]]
  #v(-1em * 1/3)
  #block[#text(weight: 500, 1.0em, document.author.join(", "))]
  #v(+1em * 1/2)
]

#set table(
  fill: (_, y) => if calc.rem(y, 2) == 0 { black.lighten(95%) },
  stroke: (x, y) => if x != 0 and y != 0 { (left: 0.05em + black, top: 0.05em + black) } else if x != 0 { (left: 0.05em + black) } else if y != 0 { (top: 0.05em + black) },
)

#set math.equation(numbering: "(1)")

= Communication

We have established a WhatsApp group for communication among the project members and decided to occasionally meet in person on campus. The code is shared using a GitHub repository --- #underline[https://github.com/nedelamatej/adaml-kul].

= Exploratory Analysis

The _NASA Turbofan Jet Engine_ dataset consists of four subsets (`FD001` -- `FD004`) each with a training set, a test set and a vector of true remaining useful life (RUL) values, distinguished by operating conditions and fault modes, as shown in @tbl:subsets. Training data were loaded into Python as four matrices (20631, 53759, 24720, 61249 observations as rows and 26 predictors as columns). All further exploratory analysis in this report focuses on `FD001` (single condition, single fault mode) for clarity. However, the same steps will also be applied to the other subsets.

#figure(
  table(
    columns: (2fr, 4fr, 10fr, 2fr, 2fr),
    align: (left, left, left, right, right),
    [*Subset*], [*Operating condition*], [*Fault modes*],                             [*Units*], [*Rows*],
    [`FD001`],  [Single (sea level)],    [Single (HPC degradation)],                  [100],     [20631],
    [`FD002`],  [Six (discrete)],        [Single (HPC degradation)],                  [260],     [53759],
    [`FD003`],  [Single (sea level)],    [Two (HPC degradation and fan degradation)], [100],     [24720],
    [`FD004`],  [Six (discrete)],        [Two (HPC degradation and fan degradation)], [249],     [61249],
  ),
  caption: [_NASA Turbofan Jet Engine_ dataset subsets (all of them) and their characteristics.]
) <tbl:subsets>

Each subset has 26 columns, starting with unit number, time (cycle), 3 operational settings and 21 sensor measurements, as shown in @tbl:columns, with symbols and physical meaning taken from _Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation_ (Saxena et al., 2008). The 3 operational settings (altitude, Mach number, TRA) take only 6 discrete combinations and could be treated as a single categorical attribute for operating condition.

It has been verified that there are no missing values and that each unit's time series is continuous without any gaps. However, the units have varying lifetimes, as shown in histogram in @fig:histogram. Columns `T2`, `P2`, `Nf_dmd` and `PCNfR_dmd` have zero standard deviation within each operating condition and therefore are redundant for modelling. Other sensor scales differ by orders of magnitude and require normalization or standardization before modelling.

#figure(
  image("figs/histogram.svg"),
  caption: [Histogram of unit lifetimes for `FD001`.],
) <fig:histogram>

Given the two fault modes (HPC and fan degradation), sensors expected to be most informative for RUL include `T30`, `P30`, `Ps30` (HPC outlet conditions), `Nf` and `NRf` (fan speed). Summary statistics and plots for the `FD001` HPC outlet sensors are shown in @tbl:predictors and @fig:predictors. It can be observed that the total temperature and static pressure at HPC outlet rise over the unit's lifetime.

#figure(
  table(
    columns: (2fr, 8fr, 2fr, 2fr, 2fr, 2fr, 2fr),
    align: (left, left, left, right, right, right, right),
    [*Symbol*], [*Description*],                   [*Unit*],  [*Mean*], [*Std. dev.*],  [*Min.*],  [*Max.*],
    [`T30`],    [Total temperature at HPC outlet], [°R],     [1590.52],        [6.13], [1571.04], [1616.91],
    [`P30`],    [Total pressure at HPC outlet],    [psia],    [553.37],        [0.89],  [549.85],  [556.06],
    [`Ps30`],   [Static pressure at HPC outlet],   [psia],     [47.54],        [0.27],   [46.85],   [48.53],
  ),
  caption: [_NASA Turbofan Jet Engine_ dataset predictors (three of them) and their statistics.],
) <tbl:predictors>

#figure(
  grid(
    columns: (1fr, 1fr),
    image("figs/total_temperature.svg"),
    image("figs/static_pressure.svg"),
  ),
  caption: [Total temperature (`T30`) and static pressure (`Ps30`) at HPC outlet for `FD001`.],
) <fig:predictors>

= Principal Component Analysis

The principal component analysis was applied to all of the four subsets. Before applying the principal component analysis on the dataset, all sensor features were standardized to eliminate the scale difference and to ensure equal contribution to the variance calculation. After applying the principal component analysis, we took 2 principal component (PC) which reduced the high-dimensional complexity of the dataset but captured a substantial share of the total variance, though considerably more for `FD002` and `FD004` (both 97 %) than for `FD001` and `FD003` (74 % and 76 %, respectively). The high variance captured for `FD002` and `FD004` likely reflects the added variation from six operating conditions rather than degradation alone, motivating the per-condition normalization planned below.

The biplots (@fig:biplots_before) and loading plots (@fig:loading_plots) clearly visualize the group of 21 sensors into distinct clusters based on their behaviors telling us which physical parameters (such as temperatures, pressures and speeds) share strong relation. Comparing the subsets highlights how operating environments change engine behavior. Datasets like `FD001` and `FD003` show smooth and unified degradation trajectories because they operate under constant conditions whereas `FD002` and `FD004` show multi-branched or scattered clusters due to varying altitude, Mach number and throttle settings. The loading vectors map out physical subsystems behavior, closely aligned sensor arrows (such as specific temperature and pressure pairings) indicating that thermal and aerodynamic subsystems tend to degrade together.

#figure(
  image("figs/biplots_before.png"),
  caption: [PCA biplots before data pretreatment for all _NASA Turbofan Jet Engine_ subsets.],
) <fig:biplots_before>

#figure(
  image("figs/loading_plots.png"),
  caption: [PCA loading plots before data pretreatment for all _NASA Turbofan Jet Engine_ subsets.],
) <fig:loading_plots>

= Pretreatment

All sensor variables will be standardized before further analysis, and the four columns with zero variance (`T2`, `P2`, `Nf_dmd`, `PCNfR_dmd`) will be removed. No missing or extreme values were found. For `FD002` and `FD004`, sensor readings will additionally be normalized per operating condition before modelling, to separate degradation from changes in operating point.

= Data Pretreatment

Data pretreatment builds on the findings from the exploratory data analysis. It starts with removing noise from the 3 operational setting values and replacing these columns with one categorical variable representing the operational conditions. The lookup table of operational conditions is provided in @tbl:operating_conditions. Operational settings are dropped from the dataset after this transformation, as the information is no longer needed and is preserved in the new categorical variable.

#figure(
  table(
    columns: (5fr, 3fr, 3fr, 3fr, 6fr),
    align: (left, right, right, right, left),
    [*Operating condition*], [*Altitude \[K ft\]*], [*Mach number*],   [*TRA*], [*Note*],
    [`Op. cond. 0`],                       [$0.0$],        [$0.00$], [$100.0$], [_All subsets `FD001` -- `FD004`_],
    [`Op. cond. 1`],                      [$10.0$],        [$0.25$], [$100.0$], [_Only subsets `FD002` and `FD004`_],
    [`Op. cond. 2`],                      [$20.0$],        [$0.70$], [$100.0$], [_Only subsets `FD002` and `FD004`_],
    [`Op. cond. 3`],                      [$25.0$],        [$0.62$],  [$60.0$], [_Only subsets `FD002` and `FD004`_],
    [`Op. cond. 4`],                      [$35.0$],        [$0.84$], [$100.0$], [_Only subsets `FD002` and `FD004`_],
    [`Op. cond. 5`],                      [$42.0$],        [$0.84$], [$100.0$], [_Only subsets `FD002` and `FD004`_],
  ),
  caption: [Lookup table of operating conditions.],
) <tbl:operating_conditions>

After the operational conditions have been encoded as categorical variables, the remaining useful life (RUL) for each engine is calculated as shown in @eq:remaining_useful_life, where $u$ denotes the unit index, $t$ the cycle number, and $t_"max" (u)$ the unit's last recorded cycle. Remaining useful life is capped at 125 cycles to avoid excessively large values. Raw, uncapped RUL values are also stored for reference.

$
  "RUL"(u, t) = min(t_"max" (u) - t, 125)
$ <eq:remaining_useful_life>

As can be seen in @fig:violin_plot_before, the distribution of key sensors before standardization shows significant variability and magnitude differences across different sensors. Closer investigation of individual sensors (e.g. `Ps30` in @fig:box_plot) reveals that measurements vary significantly across different operating conditions and are therefore clearly clustered into distinct groups corresponding to the operating condition.

#figure(
  image("figs/violin_plot_before.svg"),
  caption: [Violin plot of key sensors (`T30`, `P30`, `Ps30`, `Nf` and `NRf`) before standardization.],
) <fig:violin_plot_before>

#figure(
  image("figs/box_plot.svg"),
  caption: [Box plot of static pressure at HPC outlet (`Ps30`) by operating condition.],
) <fig:box_plot>

The need for standardization per operational condition is evident from the graphs above. Standardization normalizes the data, making it more suitable for subsequent analysis and modeling. It also removes the external variability introduced by different operating conditions, allowing the fault characteristics to be represented more accurately. For standardization, the z-score method is applied, where $mu$ and $sigma$ are computed separately for each operational condition and then applied to the corresponding sensor measurements, as shown in @eq:z-score. Predictors with standard deviation close to zero ($overline(sigma) < 10^(-3)$, averaged across operational conditions) are dropped, as they provide no useful information.

$
  x' = (x - mu) / sigma
$ <eq:z-score>

Post-standardization, extreme values ($|z| > 5$) were found in less than 1.4 % of observations across all subsets. Given their small proportion and consistency with late-life degradation, no observations were removed.

Effect of standardization can be seen in @fig:violin_plot_after, where distinct separate clusters corresponding to different operational conditions are no longer present. The values are now centered around zero, as expected from the z-score standardization.

#figure(
  image("figs/violin_plot_after.svg"),
  caption: [Violin plot of key sensors (`T30`, `P30`, `Ps30`, `Nf` and `NRf`) after standardization.],
) <fig:violin_plot_after>

@fig:bypass_ratio highlights the effect of standardization on bypass ratio (`BPR`) for two engine units in `FD004`. The two units show visibly different degradation trajectories, which may reflect the presence of two distinct fault modes in `FD004`; however, the dataset provides no per-unit fault mode label to confirm this.

#figure(
  image("figs/bypass_ratio.png"),
  caption: [Bypass ratio (`BPR`) for `FD004` before standardization (on the left) and after standardization (on the right).],
) <fig:bypass_ratio>

Finally, the effect of standardization on the PCA biplots can be observed in @fig:biplots_after. In comparison with @fig:biplots_before, there are no longer separable clusters (in case of `FD002` and `FD004` which contain multiple operational conditions). The score points in FD001 and FD003 show similar shape and spread, as do those in FD002 and FD004, indicating that standardization has effectively removed the influence of different operating conditions. The subsets with multiple fault modes (`FD003` and `FD004`) also show a visible split in the score trajectories, which may reflect the two underlying fault modes, although this cannot be confirmed without ground-truth labels.

#figure(
  image("figs/biplots_after.png"),
  caption: [PCA biplots after data pretreatment for all _NASA Turbofan Jet Engine_ subsets.],
) <fig:biplots_after>

#pagebreak()

= Modelling Plan

== Modelling Goal

The purpose of designing modelling is to achieve such a model that is well trained on the train data of _NASA Turbofan Jet Engine_ dataset. The model is expected to predict on the test data set about the remaining life cycle of the corresponding engine. The final model should be built in such a way that cross-validation $Q^2 > 0.6$ and minimizing the prediction error so that the error remains within the operational tolerance limits of the engine's RUL.

== Model Calibration Strategy

In the beginning of the calibration process, to eliminate the multicollinearity and to find out the LV (latent variable) we apply PLS (partial least squares). In this method, we decompose the input matrix $X$ and output matrix $Y$ simultaneously. Then, by identifying the relation between input score $T$ and output score $U$, the covariance of the input and output is maximized. Eventually, by using the internal mechanism the raw loadings $P$ can be optimized in weight matrix $W$, which acts as the final calibration of the model.

The _NASA Turbofan Jet Engine_ training data is used as training data set. In this data set, there are different operational settings (input variables) and target output (RUL file) helps model to learn the relation between them and predict afterwards. Separate PLS models will be calibrated for each of the four subsets (`FD001` -- `FD004`), since they differ in operating conditions and fault modes.

== Model Validation Strategy

The initial scope of the validation ensures that model is not overfitting or underfitting meaning that even outside of the training data model learns well and predict the output with mentioned accuracy. Also, deciding the number of latent variables is part of the validation. Too many LVs may lead to local model and too few LVs may not capture relevant variations. Another validation point would be to verify if the model works well for small or large values.

To tune the internal parameters of the model, and for cross validation k-fold cross-validation will be used. In our validation methodology the below steps will be followed:

+	For choosing correct number of PLS component, first validation will be done with cross-validation.
+	In next step, 10%--25% of the total sample will be left out and form groups on the remaining samples. Then each sample groups will be ruled out only once.
+	The left-out group sample will be predicted by the model. In this way, on the basis of the summation of prediction error or maximum of $Q^2$ value, number of minimum error is chosen as number of LVs.

== Model Testing Strategy

To test the model, $"RMSE"$ and $R^2$ value will be used to determine the error of the model. NASA Turbofan test data set will be used to determine the model prediction which is completely independent of the training data. As a result, the model will not be biased by the training data.

== Mathematical Methods Used in Modelling

The following mathematical methods will be used in modelling:

- *Bi-linear decomposition:* It will decompose the input matrix $X$ and output matrix $Y$ into score matrix $T, U$ and loadings matrix $P, Q$ and in residual $E, F$ respectively, as can be seen in @eq:bilinear_decomposition.

$
  X &= T P^T + E \
  Y &= U Q^T + F
$ <eq:bilinear_decomposition>

- *Score relation and co-variance maximization:* It will establish the relationship between score matrix of input and output which is $T$ and $U$ and maximize the covariance between input and output, as can be seen in @eq:score_relation.

$
  U &= T d + Q
$ <eq:score_relation>

- *Regression coefficients:* Using the weight matrix $W$, and loading matrices $P$ and $Q$, final regression coefficients will be calculated which will be used with test data set to predict the RUL of the engine, as can be seen in @eq:regression_coefficients.

$
  b_"pls" = W (P^T W)^(-1) Q^T
$ <eq:regression_coefficients>

#pagebreak()

= Model Diagram

Model diagram is shown in the @fig:flowchart below.

#figure(
  image("flowchart.png"),
  caption: [Flowchart of the model development process.],
) <fig:flowchart>

#pagebreak()

= Appendix

#figure(
  table(
    columns: (2fr, 8fr, 10fr),
    align: (left, left, left),
    [*Symbol*],    [*Description*],                   [*Unit*],
    [`unit_id`],   [Unit number],                     [---],
    [`cycle`],     [Time],                            [cycle],
    [`altitude`],  [Altitude],                        [K ft],
    [`mach`],      [Mach number],                     [---],
    [`throttle`],  [Throttle resolver angle],         [---],
    [`T2`],        [Total temperature at fan inlet],  [°R],
    [`T24`],       [Total temperature at LPC outlet], [°R],
    [`T30`],       [Total temperature at HPC outlet], [°R],
    [`T50`],       [Total temperature at LPT outlet], [°R],
    [`P2`],        [Pressure at fan inlet],           [psia],
    [`P15`],       [Total pressure in bypass-duct],   [psia],
    [`P30`],       [Total pressure at HPC outlet],    [psia],
    [`Nf`],        [Physical fan speed],              [rpm],
    [`Nc`],        [Physical core speed],             [rpm],
    [`epr`],       [Engine pressure ratio (P50/P2)],  [---],
    [`Ps30`],      [Static pressure at HPC outlet],   [psia],
    [`phi`],       [Ratio of fuel flow to Ps30],      [pps/psi],
    [`NRf`],       [Corrected fan speed],             [rpm],
    [`NRc`],       [Corrected core speed],            [rpm],
    [`BPR`],       [Bypass ratio],                    [---],
    [`farB`],      [Burner fuel-air ratio],           [---],
    [`htBleed`],   [Bleed enthalpy],                  [---],
    [`Nf_dmd`],    [Demanded fan speed],              [rpm],
    [`PCNfR_dmd`], [Demanded corrected fan speed],    [rpm],
    [`W31`],       [HPT coolant bleed],               [lbm/s],
    [`W32`],       [LPT coolant bleed],               [lbm/s],
  ),
  caption: [_NASA Turbofan Jet Engine_ dataset columns (all of them) and their description.]
) <tbl:columns>
