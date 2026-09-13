// [ADAML] Advanced Data Analysis and Machine Learning
//         Project 1a
//
// Lappeenranta-Lahti University of Technology
// School of Engineering Science
//
// Name: proj1a.typ
// Aut.: Moriom Akter
//       Matej Nedela
//       Monowarul Sabbir
// Date: 13/09/2026
// Ver.: 1.0

#set text(size: 9.5pt)
#set page(margin: 1.25cm)

#set document(
  title: "Advanced Data Analysis and Machine Learning (Project 1a)",
  author: ("Moriom Akter", "Matěj Neděla", "Monowarul Sabbir"),
  date: datetime(year: 2026, month: 9, day: 13),
)

#context align(center)[
  #block[#text(weight: 700, 1.6em, document.title)]
  #block[#text(weight: 500, 1.0em, document.author.join(", "))]
  #v(1em)
]

#set table(
  fill: (_, y) => if calc.rem(y, 2) == 0 { black.lighten(95%) },
  stroke: (x, y) => if x != 0 and y != 0 { (left: 0.05em + black, top: 0.05em + black) } else if x != 0 { (left: 0.05em + black) } else if y != 0 { (top: 0.05em + black) },
)

= Communication

We have established a WhatsApp group for communication among the project members and decided to occasionally meet in person on campus. The code is shared using a GitHub repository --- #underline[https://github.com/nedelamatej/adaml-kul].

= Exploratory Analysis

The _NASA Turbofan Jet Engine_ dataset consists of four subsets (`FD001` -- `FD004`) each with a training set, a test set and a vector of true remaining useful life (RUL) values, distinguished by operating conditions and fault modes, as shown in @tbl:subsets. Training data were loaded into Python as four matrices (20631, 53759, 24720, 61249 observations as rows and 26 predictors as columns). All further analysis in this report focuses on `FD001` (single condition, single fault mode) for clarity. However, the same steps will also be applied to the other subsets.

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

Each subset has 31 columns, starting with unit number, time (cycle), 3 operational settings and 26 sensor measurements, as shown in @tbl:columns, with symbols and physical meaning taken from _Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation_ (Saxena et al., 2008). The 3 operational settings (altitude, Mach number, TRA) take only 6 discrete combinations and could be treated as a single categorical attribute for operating condition.

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

TODO

= Pretreatment

TODO

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
    [`tra`],       [Throttle resolver angle],         [---],
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
