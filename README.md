# ontotype

Code for constructing "ontotypes" as defined in

Yu MK et al. Translation of Genotype to Phenotype by a Hierarchy of Cell Subsystems. Cell Syst. 2016 Feb 24;2(2):77-88.
DOI: http://dx.doi.org/10.1016/j.cels.2016.02.003

The Jupyter notebook Make_Ontotypes.ipynb contains example code and explanations for converting a set of genotypes into ontotypes.

In Yu MK et al., we regressed ontotypes to phenotypes using a supervised machine learning technique known as Random Forests. To do this same type of regression, we recommend the Python sklearn implementation of Random Forests, or a faster version of this implementation we used at https://github.com/michaelkyu/scikit-learn-fasterRF