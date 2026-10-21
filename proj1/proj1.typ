// [ADAML] Advanced Data Analysis and Machine Learning
//         Project 1
//
// Lappeenranta-Lahti University of Technology
// School of Engineering Science
//
// Name: proj1.typ
// Aut.: Moriom Akter
//       Matej Nedela
//       Monowarul Sabbir
// Date: 22/10/2026
// Ver.: 1.0

#set text(size: 9.5pt)
#set page(margin: 1.25cm)

#set document(
  title: "Advanced Data Analysis and Machine Learning (Project 1)",
  author: ("Moriom Akter", "Matěj Neděla", "Monowarul Sabbir"),
  date: datetime(year: 2026, month: 10, day: 22),
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
    [*Subset*], [*Operating condition*], [*Fault modes*],                             [*Units*],  [*Rows*],
    [`FD001`],  [Single (sea level)],    [Single (HPC degradation)],                      [100],   [20631],
    [`FD002`],  [Six (discrete)],        [Single (HPC degradation)],                      [260],   [53759],
    [`FD003`],  [Single (sea level)],    [Two (HPC degradation and fan degradation)],     [100],   [24720],
    [`FD004`],  [Six (discrete)],        [Two (HPC degradation and fan degradation)],     [249],   [61249],
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
    [*Symbol*], [*Description*],                   [*Unit*],    [*Mean*], [*Std. dev.*],  [*Min.*],  [*Max.*],
    [`T30`],    [Total temperature at HPC outlet], [°R],       [1590.52],        [6.13], [1571.04], [1616.91],
    [`P30`],    [Total pressure at HPC outlet],    [psia],      [553.37],        [0.89],  [549.85],  [556.06],
    [`Ps30`],   [Static pressure at HPC outlet],   [psia],       [47.54],        [0.27],   [46.85],   [48.53],
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

The principal component analysis was applied to all four subsets. Before applying the principal component analysis on the dataset, all sensor features were standardized to eliminate the scale difference and to ensure equal contribution to the variance calculation. After applying the principal component analysis, we took 2 principal components (PC) which reduced the high-dimensional complexity of the dataset but captured a substantial share of the total variance, though considerably more for `FD002` and `FD004` (both 97 %) than for `FD001` and `FD003` (74 % and 76 %, respectively). The high variance captured for `FD002` and `FD004` likely reflects the added variation from six operating conditions rather than degradation alone, motivating the per-condition normalization planned below.

The biplots (@fig:biplots_before) and loading plots (@fig:loading_plots) clearly visualize the group of 21 sensors into distinct clusters based on their behaviors telling us which physical parameters (such as temperatures, pressures and speeds) share strong relation. Comparing the subsets highlights how operating environments change engine behavior. Datasets like `FD001` and `FD003` show smooth and unified degradation trajectories because they operate under constant conditions whereas `FD002` and `FD004` show multi-branched or scattered clusters due to varying altitude, Mach number and throttle settings. The loading vectors map out physical subsystems behavior, closely aligned sensor arrows (such as specific temperature and pressure pairings) indicating that thermal and aerodynamic subsystems tend to degrade together.

#figure(
  image("figs/biplots_before.png"),
  caption: [PCA biplots before data pretreatment for all _NASA Turbofan Jet Engine_ subsets.],
) <fig:biplots_before>

#figure(
  image("figs/loading_plots.png"),
  caption: [PCA loading plots before data pretreatment for all _NASA Turbofan Jet Engine_ subsets.],
) <fig:loading_plots>

= Data Pretreatment

Compared to the initial plan, standardization per operational condition is applied to all subsets, not only `FD002` and `FD004`. Predictors are dropped based on their standard deviation rather than a fixed list, which additionally removes `epr` and `farB`, leaving 15 of 21 sensors. Remaining useful life capping and rolling mean smoothing have also been added.

Data pretreatment starts with rounding the 3 operational settings to remove noise and replacing these columns with one categorical variable representing the operational conditions. The lookup table of operational conditions is provided in @tbl:operating_conditions. Operational settings are dropped from the dataset after this transformation, as the information is no longer needed and is preserved in the new categorical variable.

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

After the operational conditions have been encoded as categorical variables, the remaining useful life (RUL) for each engine is calculated as shown in @eq:remaining_useful_life, where $u$ denotes the unit index, $t$ the cycle number, and $t_"max" (u)$ the unit's last recorded cycle. Remaining useful life is capped at 125 cycles. Early in its life, an engine operates in a healthy state and degradation is not yet observable in the sensor measurements (see @fig:predictors and @fig:bypass_ratio), so the data cannot distinguish RUL 200 from RUL 150. We focused on accurate predictions at low RUL, which are more relevant from a maintenance perspective, so a capped value is interpreted as "healthy". Model predictions are therefore clipped to $[0, 125]$ and a prediction of $125$ means healthy state with RUL $>= 125$. The true RUL of the test units is capped in the same way for evaluation.

$
  "RUL"(u, t) = min(t_"max" (u) - t, 125)
$ <eq:remaining_useful_life>

As can be seen in @fig:violin_plot_before, the distribution of key sensors before standardization shows significant variability and magnitude differences across sensors. Closer investigation of individual sensors (e.g. `Ps30` in @fig:box_plot) reveals that measurements vary significantly across different operating conditions and are therefore clustered into distinct groups corresponding to the operating condition.

#figure(
  image("figs/violin_plot_before.svg"),
  caption: [Violin plot of key sensors (`T30`, `P30`, `Ps30`, `Nf` and `NRf`) before standardization.],
) <fig:violin_plot_before>

#figure(
  image("figs/box_plot.svg"),
  caption: [Box plot of static pressure at HPC outlet (`Ps30`) by operating condition.],
) <fig:box_plot>

The need for standardization per operational condition is evident from the graphs above. Standardization normalizes the data, making it more suitable for subsequent analysis and modeling. It also removes the external variability introduced by different operating conditions, allowing the fault characteristics to be represented more accurately. For standardization, the z-score method is applied, as shown in @eq:z-score, where $mu_(j, c)$ and $sigma_(j, c)$ are the mean and standard deviation of sensor $j$ under operational condition $c$, and $c$ is the operational condition of observation $i$. These parameters are estimated from the training data of all subsets combined and then applied to both training and test data. Predictors whose standard deviation falls below $10^(-3)$ within any combination of subset and operational condition are dropped, as they provide no useful information.

$
  x'_(i, j) = (x_(i, j) - mu_(j, c)) / sigma_(j, c)
$ <eq:z-score>

Post-standardization, extreme values ($|z| > 5$) were found in less than 1.4 % of observations across all subsets. Given their small proportion and consistency with late-life degradation, no observations were removed.

Effect of standardization can be seen in @fig:violin_plot_after, where distinct clusters corresponding to different operational conditions are no longer present. The values are now centered around zero, as expected from the z-score standardization.

#figure(
  image("figs/violin_plot_after.svg"),
  caption: [Violin plot of key sensors (`T30`, `P30`, `Ps30`, `Nf` and `NRf`) after standardization.],
) <fig:violin_plot_after>

@fig:bypass_ratio highlights the effect of standardization on bypass ratio (`BPR`) for two engine units in `FD004`. The two units show visibly different degradation trajectories, which may reflect the presence of two distinct fault modes in `FD004`; however, the dataset provides no per-unit fault mode label to confirm this.

#figure(
  image("figs/bypass_ratio.png"),
  caption: [Bypass ratio (`BPR`) for `FD004` before standardization (on the left) and after standardization (on the right).],
) <fig:bypass_ratio>

After standardization, a trailing rolling mean with a window of 10 cycles is applied to each sensor within each unit to reduce measurement noise. Only the current and previous cycles of the same unit are used (for the first cycles the window is shortened). The smoothing is applied identically to the test data.

#figure(
  image("figs/biplots_after.png"),
  caption: [PCA biplots after data pretreatment for all _NASA Turbofan Jet Engine_ subsets.],
) <fig:biplots_after>

Finally, the effect of standardization on the PCA biplots can be observed in @fig:biplots_after. In comparison with @fig:biplots_before, there are no longer separable clusters (in case of `FD002` and `FD004` which contain multiple operational conditions). The score points in `FD001` and `FD003` show similar shape and spread, as do those in `FD002` and `FD004`, indicating that standardization has effectively removed the influence of different operating conditions. The subsets with multiple fault modes (`FD003` and `FD004`) also show a visible split in the score trajectories, which may reflect the two underlying fault modes, although this cannot be confirmed without ground-truth labels.

= Modelling Plan

== Modelling Goal

The goal is to calibrate a soft-sensor to estimate the remaining useful life (RUL) maintenance due time for turbofan engines using the _NASA Turbofan Jet Engine_ dataset. It is hard to set rigid KPIs for the $Q^2$ metric or operational tolerance limits before modelling, as we do not know the exact operational tolerance limits of the engine's RUL. Moreover, if our model goes off at high-RUL values (has a large residual for high RUL -- such as predicting an RUL of 120 when in reality it is 95) and predicts low-RUL very well with a very small residual, it can be considered as a good model. When an engine is close to maintenance (has a small RUL) the information is much more useful and critical.

== Model Calibration Strategy

To calibrate the model, tools and methodology are applied using the _NASA Turbofan Jet Engine_ training dataset. Since we have one response variable ($Y = "RUL"$) and not a multivariate $Y$, a univariate version of partial least squares (PLS) is used. To evaluate the calibration, $R^2_"cal" (X)$ and $R^2_"cal" (Y)$ can be used. The matrices presented in the calibration strategy are placed next to their respective equations:

*Bi-linear decomposition:* Decomposes the input matrix ($X$) and output matrix ($Y$) into score matrices ($T$, $U$), loading matrices ($P$, $Q$) and residuals ($E$, $F$) respectively, as can be seen in @eq:bilinear_decomposition.

$
  X &= T dot P^T + E \
  Y &= U dot Q^T + F
$ <eq:bilinear_decomposition>

*Score relation and covariance maximization:* Establishes the relationship between the score matrix of input ($T$) and output ($U$) to maximize covariance, as can be seen in @eq:score_relation.

$
  U &= T dot d + G
$ <eq:score_relation>

Here $d$ is the scaling/regression coefficient linking the score components and $G$ is the residual matrix.

*Regression coefficients:* Using the weight matrix ($W$) and loading matrices ($P$ and $Q$) final regression coefficients are calculated, which will be used with the test dataset to predict the RUL of the engine, as can be seen in @eq:regression_coefficients.

$
  b_"pls" &= W dot (P^T dot W)^(-1) dot Q^T
$ <eq:regression_coefficients>

== Model Validation Strategy

The initial scope of the validation ensures that model is not overfitting or underfitting meaning that even outside of the training data model learns well and predict the output with accuracy. Also, deciding the number of latent variables (LV) is part of the validation. Too many LVs may lead to local model and too few LVs may not capture relevant variations. Another validation point would be to verify if the model works well for small or large values.

To tune the internal parameters, cross validation is used. The validation strategy follows below order of steps:

- When performing k-folds, it will be made sure that the units (entire engine units as small time-series) are divided and not individual samples. The units stay together in a bucket, ensuring that consecutive cycles of an engine do not get split across different folds.
- Leave out 10 %--25 % of the total units to form groups on the remaining samples, ensuring each sample group is left out only once.
- The left-out group samples are predicted by the model.
- On the basis of the summation of prediction error or maximum $Q^2$ value, the number of LVs with minimum error is chosen.

== Model Testing Strategy

To test the model, $"RMSE"$ and $R^2$ value can be used to determine the error and predictive performance of the model. We can also analyze the residual and how they vary as a function of the RUL. $R^2$ is a metric to evaluate the goodness of fit for the test data. The _NASA Turbofan Jet Engine_ test dataset is completely independent of the training data, ensuring the model remains unbiased toward the train set.

== Mathematical Methods Used in Modelling

Bi-linear decomposition decomposes the input matrix ($X$) and output matrix ($Y$) into score matrices ($T$, $U$), loading matrices ($P$, $Q$) and residuals ($E$, $F$) as $X = T dot P^T + E$ and $Y = U dot Q^T + F$. Score relation and covariance maximization establish the relationship between the score matrix of input ($T$) and output ($U$), maximizing the covariance through $U = T dot d + G$, where $d$ represents the scaling coefficient and $G$ is the residual. Finally, regression coefficients are calculated using the weight matrix ($W$) and loading matrices ($P$ and $Q$) via $b_"pls" = W dot (P^T dot W)^(-1) dot Q^T$ to predict the RUL on the new data. A univariate version of PLS is utilized because there is only one response variable ($Y = "RUL"$).

#pagebreak()

= Model Diagram

Model diagram is shown in @fig:flowchart below.

#figure(
  image("flowchart.png"),
  caption: [Flowchart of the model development process.],
) <fig:flowchart>

= Initial Model (Partial Least Squares)

The implementation uses `PLSRegression` which is linear PLS. In our _NASA Turbofan Jet Engine_ dataset file we have 4 different subsets, all of them containing train dataset, test dataset and corresponding RUL for the test dataset, with different operational settings and failure condition. After the data pretreatment, we have used $"VIP"$ and cross-validated $R^2$ of individual sensors to select important variables. A sensor is kept if its VIP averaged across subsets is above $0.5$ (@fig:vip_score) and its average $R^2$ is above $0$ (@fig:r2_score). All 15 sensors satisfied both criteria, so no further sensors were removed. After that, we have built a correlation heatmap with the selected variables (@fig:correlation_heatmap), where it can be seen that sensors in `FD001` and `FD002` are strongly correlated with each other, while in `FD003` and `FD004` the correlations are weaker (e.g. `BPR`, `W31` and `W32`), which may reflect the two different fault modes affecting different sensors.

The initial results of our four trained partial least squares models ($"RMSE"$ and $R^2$) are shown in @tbl:results.

#figure(
  table(
    columns: (2fr, 4fr, 6fr, 4fr, 4fr),
    align: (left, left, left, right, right),
    [*Subset*], [*Operating condition*], [*Fault modes*],               [*$"RMSE"$*], [*$R^2$*],
    [`FD001`],  [Single (sea level)],    [Single (HPC degradation)],       [$21.72$],  [$0.71$],
    [`FD002`],  [Six (discrete)],        [Single (HPC degradation)],       [$20.46$],  [$0.77$],
    [`FD003`],  [Single (sea level)],    [Two (HPC deg. and fan deg.)],    [$20.88$],  [$0.72$],
    [`FD004`],  [Six (discrete)],        [Two (HPC deg. and fan deg.)],    [$21.95$],  [$0.74$],
  ),
  caption: [Initial PLS models results ($"RMSE"$, $R^2$) for each of the subsets.]
) <tbl:results>

#figure(
  image("figs/vip_score.svg"),
  caption: [Variable importance in projection (VIP) across all subsets (`FD001` -- `FD004`).],
) <fig:vip_score>

#figure(
  image("figs/r2_score.svg"),
  caption: [$R^2$ cross-validation across all subsets (`FD001` -- `FD004`).],
) <fig:r2_score>

#figure(
  image("figs/correlation_heatmap.svg"),
  caption: [Sensor measurements correlation matrices for each subset.],
) <fig:correlation_heatmap>

The number of latent variables was selected based on $Q^2$ cross-validation (@fig:q2_heatmap) as the smallest number of LVs whose $Q^2$ is within $0.01$ of the maximum. This resulted in 2, 1, 3 and 3 LVs for `FD001` -- `FD004`, respectively. The selected LVs correspond to $Q^2$ values between $0.74$ and $0.78$ for all subsets, which indicates good predictive ability. Adding more LVs does not improve $Q^2$ noticeably. The $R^2$ on the test data (@fig:r2_heatmap) is shown only for comparison and to check whether the models are overfitting or not. Similarly to $Q^2$, the $R^2$ values span from $0.71$ to $0.77$ for the selected number of LVs, which is close to $Q^2$, indicating that the models are not overfitting.

#figure(
  image("figs/q2_heatmap.svg"),
  caption: [Initial PLS models $Q^2$ performance across all subsets based on number of LVs (red box shows chosen LVs).],
) <fig:q2_heatmap>

#figure(
  image("figs/r2_heatmap.svg"),
  caption: [Initial PLS models $R^2$ performance across all subsets based on number of LVs (red box shows chosen LVs).],
) <fig:r2_heatmap>

@fig:predicted_vs_observed shows the comparison of predicted and true remaining useful life. Our trained models follow the diagonal (highlighted in red) reasonably well. As previously stated, an RUL of $125$ corresponds to RUL $>= 125$, meaning a healthy state of the engine unit. For these units the predictions in `FD002` and `FD004` spread between approx. $75$ and $125$. Overall, our models tend to overestimate the RUL mainly for RUL lower than $80$, which is worse than underestimating it, as the maintenance could be planned too late.

#figure(
  image("figs/predicted_vs_observed.svg"),
  caption: [Initial PLS models predicted vs observed remaining useful life across all subsets (`FD001` -- `FD004`).],
) <fig:predicted_vs_observed>

Residual analysis shown in @fig:residual_analysis provides similar information as @fig:predicted_vs_observed, showing that the residuals are within approximately $plus.minus 60$ cycles for all subsets. The diagonal lines of points in `FD002` and `FD004` are caused by the introduced RUL capping. The residuals also show a slight systematic pattern (overestimating predictions for $"RUL" in [20, 80]$), suggesting that the relation between sensor measurements and RUL is not fully linear. A non-linear PLS (e.g. kernel PLS) could therefore improve the predictions.

#figure(
  image("figs/residual_analysis.svg"),
  caption: [Initial PLS models residual analysis across all subsets (`FD001` -- `FD004`).],
) <fig:residual_analysis>

= Final Model (Kernel Partial Least Squares)

Kernel partial least squares with RBF kernel is introduced for the final model, as the residual analysis of the initial models suggested that the relation between sensor measurements and RUL is not fully linear. The kernel is approximated using Nyström with 300 components and $gamma = 1 / 31$ (one over the number of features), which is fitted only on the training folds in the cross-validation. Another change besides the switch to kPLS is the extension of features used. The 15 predictors are extended by the slope of each predictor, calculated as the difference of its smoothed value between cycles $t$ and $t - 10$, divided by 10 (zero for the first 10 cycles), which describes the trend of the degradation. Also the engine cycle was added, increasing the total number of features to 31. The response is converted to logarithmic scale using $y' = log(1 + y)$ before the modelling and back using $y = exp(y') - 1$ after it, which gives more weight to the low values of RUL. The predictions are clipped to $[0, 125]$ as before.

The final results of our four trained kernel partial least squares models are shown in @tbl:kpls_results. It can be seen that the $"RMSE"$ has lowered for all of the models, with the smallest improvement of $3.6$ cycles for `FD002` and the largest of $6.7$ cycles for `FD003`. The $R^2$ metric raised for all of the models as well, with the smallest improvement being $0.08$ and the largest $0.15$, reaching approximately $0.85$ or higher.

#figure(
  table(
    columns: (2fr, 4fr, 6fr, 4fr, 4fr),
    align: (left, left, left, right, right),
    [*Subset*], [*Operating condition*], [*Fault modes*],               [*$"RMSE"$*], [*$R^2$*],
    [`FD001`],  [Single (sea level)],    [Single (HPC degradation)],       [$15.50$],  [$0.85$],
    [`FD002`],  [Six (discrete)],        [Single (HPC degradation)],       [$16.90$],  [$0.85$],
    [`FD003`],  [Single (sea level)],    [Two (HPC deg. and fan deg.)],    [$14.17$],  [$0.87$],
    [`FD004`],  [Six (discrete)],        [Two (HPC deg. and fan deg.)],    [$16.90$],  [$0.85$],
  ),
  caption: [Final kPLS models results ($"RMSE"$, $R^2$) for each of the subsets.]
) <tbl:kpls_results>

The number of latent variables was selected with the exact same method as for the initial model, resulting in 9, 6, 8 and 10 LVs for `FD001` -- `FD004`, respectively. The number of LVs is higher than for the initial models, which is expected, as the Nyström approximation maps the predictors into 300 dimensions. Based on the graphs (@fig:kpls_q2_heatmap and @fig:kpls_r2_heatmap) it can be said that the models have good predictive ability and are not overfitting, as the $Q^2$ values ($0.79$ -- $0.86$) are close to the $R^2$ values on the test data ($0.85$ -- $0.87$) for the chosen LVs.

#figure(
  image("figs/kpls_q2_heatmap.svg"),
  caption: [Final kPLS models $Q^2$ performance across all subsets based on number of LVs (red box shows chosen LVs).],
) <fig:kpls_q2_heatmap>

#figure(
  image("figs/kpls_r2_heatmap.svg"),
  caption: [Final kPLS models $R^2$ performance across all subsets based on number of LVs (red box shows chosen LVs).],
) <fig:kpls_r2_heatmap>

The overestimation of remaining useful life for engines with low RUL, that could be seen in the initial model (@fig:predicted_vs_observed), was mostly solved by the changes in the final model. The predictions for RUL lower than approx. $40$ are now close to the diagonal for all subsets. The final comparison of observed remaining useful life and the value predicted by the final kPLS model can be seen below in @fig:kpls_predicted_vs_observed.

#figure(
  image("figs/kpls_predicted_vs_observed.svg"),
  caption: [Final kPLS models predicted vs observed remaining useful life across all subsets (`FD001` -- `FD004`).],
) <fig:kpls_predicted_vs_observed>

The residuals improved notably between the initial PLS models and the final kPLS models for the lower values of RUL, as can be seen in @fig:kpls_residual_analysis. However, the residuals are still quite big for the higher values of remaining useful life (up to approx. $plus.minus 60$ cycles). This makes sense, as early in the engine's life the degradation is barely visible in the sensor measurements. Same as for the initial models, the diagonal lines of points in `FD002` and `FD004` are caused by the RUL capping.

#figure(
  image("figs/kpls_residual_analysis.svg"),
  caption: [Final kPLS models residual analysis across all subsets (`FD001` -- `FD004`).],
) <fig:kpls_residual_analysis>

#pagebreak()

= References

[1] Y. Lei, N. Li, L. Guo, N. Li, T. Yan, and J. Lin, "Machinery health prognostics: A systematic review from data acquisition to RUL prediction," _Mechanical Systems and Signal Processing_, vol. 104, pp. 799--834, 2018, doi: #link("https://doi.org/10.1016/j.ymssp.2017.11.016")[10.1016/j.ymssp.2017.11.016]. \
[2] X.-S. Si, W. Wang, C.-H. Hu, and D.-H. Zhou, "Remaining useful life estimation -- A review on the statistical data driven approaches," _European Journal of Operational Research_, vol. 213, no. 1, pp. 1--14, 2011, doi: #link("https://doi.org/10.1016/j.ejor.2010.11.018")[10.1016/j.ejor.2010.11.018]. \
[3] S. Wold, M. Sjöström, and L. Eriksson, "PLS-regression: a basic tool of chemometrics," _Chemometrics and Intelligent Laboratory Systems_, vol. 58, no. 2, pp. 109--130, 2001, doi: #link("https://doi.org/10.1016/S0169-7439(01)00155-1")[10.1016/S0169-7439(01)00155-1].

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
