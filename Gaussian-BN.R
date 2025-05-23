#########################################
#####  Gaussian Bayesian Networks  ######
#########################################


install.packages("bnlearn")
install.packages("corpcor")


library(bnlearn)
library(corpcor)

dag.bnlearn <- model2network("[G][E][V|G:E][N|V][W|V][C|N:W]")
dag.bnlearn 
# checking with dsep
dsep(dag.bnlearn, "N", "W", "V")  # returns TRUE, N ⊥ W | V
# check paths
path.exists(dag.bnlearn, from = "E", to = "C")  # returns TRUE

# Defined Local Distributions
dist.list <- list(
  G = list(coef = 50, sd = 10),
  E = list(coef = 50, sd = 10),
  V = list(coef = c("(Intercept)" = -10.35534, G = 0.5, E = 0.70711), sd = 5),
  N = list(coef = c("(Intercept)" = 45, V = 0.1), sd = 9.949874),
  W = list(coef = c("(Intercept)" = 15, V = 0.7), sd = 7.141428),
  C = list(coef = c("(Intercept)" = 0, N = 0.3, W = 0.7), sd = 6.252)
)

# fit the BN Model
gbn.bnlearn <- custom.fit(dag.bnlearn, dist.list)
gbn.bnlearn

# Simulate data from the model
set.seed(123)
sim.data <- rbn(gbn.bnlearn, n = 1000)

# Ensure correct column names
colnames(sim.data) <- c("G", "E", "V", "N", "W", "C")

#########################################
### Conditional Independence Test: C ⫫ W | N
#########################################

# Compute correlation matrix and partial correlation matrix
cor.mat <- cor(sim.data)
pcor.mat <- cor2pcor(cor.mat)

# Set dimnames to match variable names in your data
var.names <- colnames(sim.data)
dimnames(pcor.mat) <- list(var.names, var.names)

# Now you can extract the partial correlation ρ_{C,W|N}
rho_CW_N <- pcor.mat["C", "W"]
print(paste("Partial correlation (C ⫫ W | N):", round(rho_CW_N, 4)))

# Compute t-statistic for hypothesis testing
n <- nrow(sim.data)
t_stat <- rho_CW_N * sqrt((n - 3) / (1 - rho_CW_N^2))
print(paste("t-statistic:", round(t_stat, 4)))

# Compute p-value from the t-distribution
p_value <- 2 * pt(-abs(t_stat), df = n - 3)
print(paste("p-value:", signif(p_value, 4)))

# Interpretation of the result
if (p_value < 0.05) {
  print("Reject H0: C and W are conditionally dependent given N.")
} else {
  print("Fail to reject H0: C and W are conditionally independent given N.")
}



#########################################
### Model Scoring
#########################################

# Compute BIC score for the model
bic_score <- score(dag.bnlearn, data = sim.data, type = "bic-g")
cat("BIC Score:", bic_score, "\n")

# Compute BGe score
bge_score <- score(dag.bnlearn, data = sim.data, type = "bge")
cat("BGe Score:", bge_score, "\n")

#########################################
### Optional: Network Visualization
#########################################

install.packages("ggplot2")
install.packages("GGally")  # for ggpairs plot
library(ggplot2)
library(GGally)


if (!requireNamespace("Rgraphviz", quietly = TRUE)) {
  BiocManager::install("Rgraphviz")
}
library(Rgraphviz)
graphviz.plot(dag.bnlearn)

# Convert data to a data frame
sim.data <- as.data.frame(sim.data)

# Histogram for each variable
for (var in colnames(sim.data)) {
  p <- ggplot(sim.data, aes_string(x = var)) +
    geom_histogram(binwidth = 2, fill = "skyblue", color = "black") +
    theme_minimal() +
    labs(title = paste("Histogram of", var), x = var, y = "Count")
  print(p)
}


# Pairwise Plot to See Relationships
ggpairs(sim.data, 
        title = "Pairwise Relationships",
        upper = list(continuous = "cor"),
        lower = list(continuous = "smooth"),
        diag = list(continuous = "densityDiag"))

# Scatter Plot: W vs C Colored by N (for Conditional Insight)
ggplot(sim.data, aes(x = W, y = C, color = N)) +
  geom_point(alpha = 0.6) +
  scale_color_viridis_c() +
  theme_minimal() +
  labs(title = "Scatter Plot of C vs W colored by N",
       x = "W", y = "C", color = "N")


#  Correlation Heatmap 
install.packages("reshape2")
library(reshape2)

cor_matrix <- round(cor(sim.data), 2)
melted_cor <- melt(cor_matrix)

ggplot(melted_cor, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "red", high = "green", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab",
                       name = "Pearson\nCorrelation") +
  geom_text(aes(label = value), color = "black", size = 4) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  labs(title = "Correlation Heatmap")