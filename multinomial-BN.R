#########################################
####  Multinomial Bayesian Networks  ####
#########################################


install.packages("bnlearn")
library(bnlearn)

# Create empty DAG with 6 nodes (no arcs yet)
dag <- empty.graph(nodes = c("A", "S", "E", "O", "R", "T"))
dag

# Method1: Adding arcs (direct dependencies)
dag <- set.arc(dag, from = "A", to = "E")  # Age → Education
dag <- set.arc(dag, from = "S", to = "E")  # Sex → Education
dag <- set.arc(dag, from = "E", to = "O")  # Education → Occupation
dag <- set.arc(dag, from = "E", to = "R")  # Education → Residence
dag <- set.arc(dag, from = "O", to = "T")  # Occupation → Travel
dag <- set.arc(dag, from = "R", to = "T")  # Residence → Travel

# Model summary
dag

modelstring(dag)  # Shows structure as conditional dependencies
nodes(dag)        # Lists all nodes
arcs(dag)         # Lists all arcs


# Method 2: Efficient Arc Setting in Bayesian Networks --> using a matrix of arcs
# Create empty graph
dag2 <- empty.graph(nodes = c("A", "S", "E", "O", "R", "T"))

# Define all arcs at once in a matrix
arc.set <- matrix(c("A", "E",
                    "S", "E",
                    "E", "O",
                    "E", "R",
                    "O", "T",
                    "R", "T"),
                  byrow = TRUE, ncol = 2,
                  dimnames = list(NULL, c("from", "to")))

# Apply arc set to DAG
arcs(dag2) <- arc.set

dag2


# Same arc as before
all.equal(dag, dag2)


# BNs are acyclic --> adding a cycle will raise an Error: the resulting graph contains cycles
set.arc(dag, from = "T", to = "E")


# Adding probability distributions for each variable
A.lv <- c("young", "adult", "old")
S.lv <- c("M", "F")
E.lv <- c("high", "uni")
O.lv <- c("emp", "self")
R.lv <- c("small", "big")
T.lv <- c("car", "train", "other")

# 1) Unconditional (1D) tables
A.prob <- array(c(0.30, 0.50, 0.20), dimnames = list(A = A.lv))
S.prob <- array(c(0.60, 0.40), dimnames = list(S = S.lv))

# 2) Conditional (2D) tables, (e.g. Occupation and Residence depend on Education)
O.prob <- array(c(0.96, 0.04, 0.92, 0.08), dim = c(2, 2),
                dimnames = list(O = O.lv, E = E.lv))

R.prob <- array(c(0.25, 0.75, 0.20, 0.80), dim = c(2, 2),
                dimnames = list(R = R.lv, E = E.lv))

# 3) 3D Conditional tables (2 parents): Education | Age, Sex
E.prob <- array(c(0.75, 0.25, 0.72, 0.28, 0.88, 0.12,
                  0.64, 0.36, 0.70, 0.30, 0.90, 0.10),
                dim = c(2, 3, 2),
                dimnames = list(E = E.lv, A = A.lv, S = S.lv))

# Travel | Occupation, Residence
T.prob <- array(c(0.48, 0.42, 0.10, 0.56, 0.36, 0.08,
                  0.58, 0.24, 0.18, 0.70, 0.21, 0.09),
                dim = c(3, 2, 2),
                dimnames = list(T = T.lv, O = O.lv, R = R.lv))


# 1. Define DAG using model formula
dag3 <- model2network("[A][S][E|A:S][O|E][R|E][T|O:R]")

# 2. Check if it matches previous DAG
all.equal(dag, dag3) # --> TRUE

# 3. Define local probability distributions
cpt <- list(A = A.prob, S = S.prob, E = E.prob, O = O.prob, R = R.prob, T = T.prob)
cpt
# 4. Create the Bayesian Network
bn <- custom.fit(dag, cpt)
bn # print all cpt in the network
# 5. Count total number of parameters
nparams(bn)
arcs(bn)


# 7. Access conditional probability table for node R
bn$R

# 8. Extract CPT for node R as an object
R.cpt <- coef(bn$R)


# dataset
survey <- read.table(text = "
A       R       E       O       S   T
adult   big     high    emp     F   car
adult   small   uni     emp     M   car
adult   big     uni     emp     F   train
adult   big     high    emp     M   car
adult   big     high    emp     M   car
adult   small   high    emp     F   train
youth   small   uni     unemp   M   bike
youth   big     high    emp     F   car
senior  small   high    unemp   F   bus
youth   small   uni     emp     M   bike
adult   big     uni     unemp   F   train
adult   small   high    emp     M   car
senior  big     high    emp     F   car
youth   big     uni     unemp   M   bike
senior  small   uni     emp     M   bus
adult   small   high    unemp   F   train
youth   small   high    emp     F   bike
adult   big     high    emp     M   car
adult   small   uni     emp     M   bus
youth   big     uni     unemp   F   bike
adult   small   high    emp     F   car
senior  big     high    emp     M   car
youth   small   uni     unemp   M   bike
adult   big     uni     emp     F   train
senior  small   uni     unemp   F   bus
youth   big     high    emp     M   car
adult   small   high    unemp   F   train
youth   small   uni     emp     F   bike
senior  big     high    emp     F   car
youth   small   high    unemp   M   bus
", header = TRUE, colClasses = "factor")

head(survey)
# Method 1: Estimating Parameters Using Maximum Likelihood
bn.mle <- bn.fit(dag, data = survey, method = "mle")
bn.mle 
bn.mle$O
# Method 2: Estimating Parameters Using Bayesian Approach
bn.bayes <- bn.fit(dag, data = survey, method = "bayes", iss = 10)
bn.bayes$O

# Conditional Independence Tests
ci.test("T", "E", c("O", "R"), test = "mi", data = survey)
# G² test (mutual information): p-value = 0.27

ci.test("T", "E", c("O", "R"), test = "x2", data = survey)
# X² test: p-value = 0.41
# --> Both p-values are large → no significant dependence → arc E → T is not supported.

ci.test("T", "O", "R", test = "x2", data = survey) # testing fir removing Arcs (e.g., O → T)
# X² test: p-value = 0.43 → arc O → T not supported


# Automating the process --> Evaluates all arcs using the specified test.
arc.strength(dag, data = survey, criterion = "x2")
# Conclusion: Arcs with p < 0.05 are well-supported.


# Comparing DAGs using Network scores (bic, bde)
score(dag, data = survey, type = "bic") 
score(dag, data = survey, type = "bde", iss = 10)
score(dag, data = survey, type = "bde", iss = 1)  
# Lower score = worse fit.
# Higher score = better fit (less negative is better).

dag4 <- set.arc(dag, from = "E", to = "T")
score(dag4, data = survey, type = "bic")
# Conclusion: Even though the likelihood improves slightly, the penalty for added complexity outweighs it. So, adding E → T is not beneficial.


# Random DAG Comparison
rnd <- random.graph(nodes = c("A", "S", "E", "O", "R", "T"))
score(rnd, data = survey, type = "bic") 
# Conclusion: A random DAG performs worse, showing the importance of informed structure learning.

# Learning a Better DAG: Hill-Climbing (hc)
learned <- hc(survey)  # Defaults to BIC
modelstring(learned)   # "[A][S][T|A][R|T][E|T][O|T]"
score(learned, data = survey, type = "bic")  # -158.8426
# Conclusion: The learned DAG fits better than all previous ones (less negative score).

# using BDe instead
learned2 <- hc(survey, score = "bde")
learned2

# arc importance: arc.strength() -> Reports how much the score drops if an arc is removed — i.e., how strong or essential the arc is.
arc.strength(learned, data = survey, criterion = "bic")
# Conclusion: Negative strength means removing the arc worsens the score (i.e., it's an important arc).

# Compare the learned DAG to original DAG
arc.strength(dag, data = survey, criterion = "bic")


# I/ Using the DAG Structure to perform d-separation tests to check conditional independence.
# 1. Serial connection: S → E → R
dsep(dag, "S", "R", "E")  # TRUE --> S and R are independent Conditioning on E, but in reality they are dependent
# 2. Divergent connection: R ← E → O
dsep(dag, "R", "O", "E")  # TRUE
# 3. Convergent connection (v-structure): A → E ← S
dsep(dag, "A", "S")        # TRUE
dsep(dag, "A", "S", "E")   # FALSE


# II/ Using Conditional Probability Tables
# 1. Exact Inference (via gRain package)
install.packages("gRain")
library(gRain)
junction <- compile(as.grain(bn)) # Convert the BN to a grain object
jsex <- setEvidence(junction, nodes = "S", states = "F") # Insert evidence with setEvidence
querygrain(jsex, nodes = "T")$T  # Query the updated distribution with querygrain
querygrain(junction, nodes = "T")$T # Query without evidence  --> Pr(T) = [car=0.5618, train=0.2809, other=0.1573]
querygrain(jsex, nodes = "T")$T  # Query with S = F ---> Result: Nearly unchanged → Sex has minimal effect on Travel preferences
jres <- setEvidence(junction, nodes = "R", states = "small") # Query with R = small
querygrain(jres, nodes = "T")$T
# --> Result: ↑train, ↓other → People in small towns prefer car/train due to limited alternatives
# Testing Conditional Independence
jedu <- setEvidence(junction, nodes = "E", states = "high")
SxT.cpt <- querygrain(jedu, nodes = c("S", "T"), type = "joint")
querygrain(jedu, nodes = c("S", "T"), type = "conditional")
SxT.ct <- SxT.cpt * nrow(survey)
chisq.test(SxT.ct) # --> Result: p-value = 1 → accept independence.

# 2. Approximate Inference (via Monte Carlo)
cpquery(bn, event = (S == "M") & (T == "car"), evidence = (E == "high")) # estimate: approximate prob
cpquery(bn, event = (S == "M") & (T == "car"),  # improve accuracy --> Result: Closer to exact value (e.g., 0.3432)
        evidence = (E == "high"), n = 10^6)


# Plotting Discrete Bayesian Networks (BNs) in R
install.packages("BiocManager")
BiocManager::install("Rgraphviz")
library(Rgraphviz)
graphviz.plot(dag) # Uses Rgraphviz to create a DAG.
# Default layout is "dot": parents above children, arcs point downward.
# Layouts can be customized: "neato", "fdp", "twopi", etc.

# Highlighting Nodes and Arcs
hlight <- list(
  nodes = nodes(dag),
  arcs = arcs(dag),
  col = "grey",       # node/edge color
  textCol = "grey"    # label color
)

pp <- graphviz.plot(dag, highlight = hlight, render = FALSE)
pp

# Change edge appearance
edgeRenderInfo(pp) <- list(
  col = c("S~E" = "black", "E~R" = "black"),
  lwd = c("S~E" = 3, "E~R" = 3)
)

# Change node appearance
nodeRenderInfo(pp) <- list(
  col = c("S" = "black", "E" = "black", "R" = "black"),
  textCol = c("S" = "black", "E" = "black", "R" = "black"),
  fill = c("E" = "grey")
)

# Render the final plot
renderGraph(pp)



# Plotting Conditional Probability Distributions (CPDs)
bn.fit.barchart(bn.mle$T, main = "Travel", xlab = "Pr(T | R, O)", ylab = "")



# Using lattice::barchart() to Compare Distributions
# Create a Data Frame of Probabilities
Evidence <- factor(c(rep("Unconditional", 3), rep("Female", 3), rep("Small City", 3)),
                   levels = c("Unconditional", "Female", "Small City"))

Travel <- factor(rep(c("car", "train", "other"), 3),
                 levels = c("other", "train", "car"))

distr <- data.frame(
  Evidence = Evidence,
  Travel = Travel,
  Prob = c(0.5618, 0.2808, 0.1573, 0.5620, 0.2806, 0.1573, 0.4838, 0.4170, 0.0990)
)

library(lattice)

barchart(Travel ~ Prob | Evidence, data = distr,
         layout = c(3, 1), xlab = "Probability",
         scales = list(alternating = 1, tck = c(1, 0)),
         strip = strip.custom(factor.levels = c(
           expression(Pr(T)),
           expression(Pr(T | S == "F")),
           expression(Pr(T | R == "small"))
         )),
         panel = function(...) {
           panel.barchart(...)
           panel.grid(h = 0, v = -1)
         })


# Combined DAG + Barchart Plots  for marginal distributions at each node
graphviz.chart(bn, grid = TRUE, main = "Original BN")

# with evidence
graphviz.chart(as.bn.fit(jedu, including.evidence = TRUE),
               grid = TRUE,
               bar.col = c(E = "grey", T = "black", ...),
               strip.bg = c(E = "grey", T = "transparent", ...),
               main = "BN with Evidence")





