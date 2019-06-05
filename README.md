# viztech
Plotnine replication of [Financial Times Visual Vocabulary](https://github.com/ft-interactive/chart-doctor/tree/master/visual-vocabulary); Inspired by [Vega](https://gramener.github.io/visual-vocabulary-vega/)

Each notebook contains `plotnine`/`ggplot` replication of the following topics, with changes we deem more sensible compared to the original approach; for instance, we DO NOT use multi-scale y-axis.
* `deviation.ipynb`: Emphasise variations (+/-) from a fixed reference point. Typically the reference point is zero but it can also be a target or a long-term average. Can also be used to show sentiment (positive/neutral/negative)
* `correlation.ipynb`: Show the relationship between two or more variables. Be mindful that, unless you tell them otherwise, many readers will assume the relationships you show them to be causal (i.e. one causes the other)
* `ranking.ipynb`: Show the relationship between two or more variables. Be mindful that, unless you tell them otherwise, many readers will assume the relationships you show them to be causal (i.e. one causes the other)
* `distribution.ipynb`: Show values in a dataset and how often they occur. The shape (or skew) of a distribution can be a memorable way of highlighting the lack of uniformity or equality in the data
* `change-over-time.ipynb`: Give emphasis to changing trends. These can be short (intra-day) movements or extended series traversing decades or centuries: Choosing the correct time period is important to provide suitable context for the reader
* `magnitude.ipynb`: Show size comparisons. These can be relative (just being able to see larger/bigger) or absolute (need to see fine differences). Usually these show a 'counted' number (for example, barrels, dollars or people) rather than a calculated rate or per cent
* `part-to-whole.ipynb`: Show how a single entity can be broken down into its component elements. If the reader's interest is solely in the size of the components, consider a magnitude-type chart instead
* `spatial.ipynb`: Used only when precise locations or geographical patterns in data are more important to the reader than anything else.
* `flow.ipynb`: Show the reader volumes or intensity of movement between two or more states or conditions. These might be logical sequences or geographical locations

We also have an original notebook specifically designed to use as data exploration tool.
* `explore.ipynb`: data visualization for exploring a dataset. The goal is to understand more about the data as a human, not to make beautiful graphs, communicate, or feature engineering input into models.

Some plots are intentionally not implemented because we think they are not good visualization practice such as pie charts and some we simply have not found an intuitive way to implement them either with `ggplot` or a simple python package yet such as sankeys, chords, networks and voronoi.
