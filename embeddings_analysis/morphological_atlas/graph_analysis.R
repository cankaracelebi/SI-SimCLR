# =============================================================================
# Morphological Transition Landscape Graph Analysis
# Input: transition probability matrix from partial optimal transport
# =============================================================================

# ── Dependencies ──────────────────────────────────────────────────────────────
required_bioc <- c("DESeq2", "ape", "limma", "geneplotter", "edgeR")
required_cran <- c("igraph", "scales", "AMR", "gprofiler2")

for (pkg in required_bioc) {
  if (!require(pkg, quietly = TRUE, character.only = TRUE))
    BiocManager::install(pkg)
  library(pkg, character.only = TRUE)
}

for (pkg in required_cran) {
  if (!require(pkg, quietly = TRUE, character.only = TRUE))
    install.packages(pkg)
  library(pkg, character.only = TRUE)
}

# ── Cluster colour palette (12 clusters) ─────────────────────────────────────
mycols <- c(
  "#2478b5", "#acc4e6",
  "#f17e21", "#f8b977",
  "#319f48", "#9bcf8a",
  "#d52e29", "#f69595",
  "#9269ab", "#c3afd4",
  "#8c5a4f", "#c39c93"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# ── Entropy metrics ───────────────────────────────────────────────────────────

#' Shannon entropy of outgoing edge weights for each node
#' Returns 0 for nodes with no outgoing edges (empty sum convention)
out_entropy <- function(g) {
  sapply(V(g), function(v) {
    e <- incident(g, v, mode = "out")
    if (length(e) == 0) return(0)
    p <- E(g)[e]$weight
    p <- p[p > 0]
    if (length(p) == 0) return(0)
    p <- p / sum(p)
    -sum(p * log(p))
  })
}

#' Shannon entropy of incoming edge weights for each node
#' Returns 0 for nodes with no incoming edges (empty sum convention)
in_entropy <- function(g) {
  sapply(V(g), function(v) {
    e <- incident(g, v, mode = "in")
    if (length(e) == 0) return(0)
    p <- E(g)[e]$weight
    p <- p[p > 0]
    if (length(p) == 0) return(0)
    p <- p / sum(p)
    -sum(p * log(p))
  })
}

#' Effective number of outgoing transitions = exp(H_out)
#' Analogous to species richness in ecology: how many equally-likely
#' targets would produce the same entropy as the observed distribution
out_eff_num <- function(g) exp(out_entropy(g))

# ── Metric visualisation ──────────────────────────────────────────────────────

#' Scatter plot of a given metric against biological condition entropy
#' with Spearman correlation annotation
plotmymetrics <- function(i) {
  plot(
    metrics[, 1], metrics[, i],
    col   = mycols, pch = 19,
    frame = FALSE, las = 1,
    xlab  = NA, ylab = NA
  )
  mtext(side = 1, line = 2, text = colnames(metrics)[1], cex = 0.7)
  mtext(side = 2, line = 2, text = colnames(metrics)[i], cex = 0.7)
  
  ct <- cor.test(metrics[, 1], metrics[, i],
                 method = "spearman", na.action = "na.omit")
  mtext(side = 3, line = 1,
        text = paste("Corr =", format(ct$estimate,
                                      scientific = FALSE, digits = 2)),
        cex = 0.7)
  mtext(side = 3, line = 0,
        text = paste("P =", format(ct$p.value,
                                   scientific = TRUE, digits = 1)),
        cex = 0.7)
}

# ── Graph visualisation with continuous colour scale ──────────────────────────

# Two colour ramps for different metric types:
# ramp1: grey -> magenta (for entropy-type metrics)
# ramp2: dark grey -> firebrick (for flow-type metrics, diverging through white)
ramp1 <- colorRamp(c(
  "grey92", "grey75", "grey55", "grey35",
  "plum2", "mediumorchid", "magenta3"
))

ramp2 <- colorRamp(c(
  "grey35", "grey55", "grey75", "grey92",
  "lightyellow", "orange", "firebrick", "darkred"
))

#' Plot transition graph with nodes coloured by a continuous metric
#' and a horizontal colour bar legend below
#'
#' @param valoi    numeric vector of node values (length = number of nodes)
#' @param valname  string label for the colour bar axis
#' @param g        igraph object
#' @param colramp  colorRamp function (default: ramp2)
PlotGraphColorsScale <- function(valoi, valname, g, colramp = ramp2) {
  
  # Replace NA with 0 (nodes with undefined metric treated as baseline)
  valoi[is.na(valoi)] <- 0
  
  # Map values to colours
  e_scaled    <- rescale(valoi, to = c(0, 1))
  rgb_mat     <- colramp(e_scaled)
  node_cols   <- rgb(rgb_mat[, 1], rgb_mat[, 2], rgb_mat[, 3],
                     maxColorValue = 255)
  V(g)$color  <- node_cols
  
  # Layout: graph panel (top) + colour bar strip (bottom)
  layout(matrix(c(1, 2), nrow = 2), heights = c(10, 1.5))
  
  # Panel 1: graph
  par(mar = c(0, 0, 2, 0))
  plot(
    g,
    layout             = layout_fixed,
    edge.arrow.size    = 0.3,
    edge.width         = E(g)$width,
    edge.curved        = 0.3,
    edge.color         = "grey30",
    vertex.label.cex   = 0.8,
    vertex.label.color = "black"
  )
  
  # Panel 2: horizontal colour bar
  par(mar = c(2.5, 5, 1, 5))
  
  n_steps    <- 200
  bar_vals   <- seq(0, 1, length.out = n_steps)
  bar_rgb    <- colramp(bar_vals)
  bar_cols   <- rgb(bar_rgb[, 1], bar_rgb[, 2], bar_rgb[, 3],
                    maxColorValue = 255)
  val_range  <- range(valoi)
  
  plot(NA,
       xlim = val_range, ylim = c(0, 1),
       xlab = "", ylab = "",
       axes = FALSE, xaxs = "i", yaxs = "i")
  
  x_breaks <- seq(val_range[1], val_range[2], length.out = n_steps + 1)
  for (i in seq_len(n_steps)) {
    rect(x_breaks[i], 0, x_breaks[i + 1], 1,
         col = bar_cols[i], border = NA)
  }
  
  box(col = "grey40", lwd = 0.8)
  axis(1,
       at       = seq(val_range[1], val_range[2], length.out = 5),
       labels   = round(seq(val_range[1], val_range[2], length.out = 5), 2),
       cex.axis = 0.75,
       col      = "grey40", col.axis = "grey20",
       lwd = 0.6, tck = -0.4)
  mtext(valname, side = 1, line = 1.8, cex = 0.8, col = "grey20")
  
  # Reset layout
  layout(1)
}

# ── Create a pruned copy for visualisation only ───────────────────────────────
prune_graph_for_viz <- function(g, quantile_thresh = 0.75) {
  
  weights   <- E(g)$weight
  threshold <- quantile(weights, quantile_thresh)
  
  # Remove edges below threshold
  g_viz <- delete_edges(g, which(weights <= threshold))
  
  cat(sprintf(
    "Visualisation graph: %d edges retained (%.0f%% removed, threshold = %.4f)\n",
    ecount(g_viz),
    100 * mean(weights <= threshold),
    threshold
  ))
  
  g_viz
}


# =============================================================================
# STATE CLASSIFICATION
# =============================================================================

#' Classify morphological substates into functional archetypes based on
#' graph-theoretic metrics derived from the transition probability matrix.
#'
#' Classification scheme:
#'   Sink/Attractor    : net receiver (var_out - var_in < -2 AND net_flow < 0)
#'   Source/Progenitor : net sender   (var_out - var_in >  2 AND net_flow > 0)
#'   Transient         : balanced     (|var_out - var_in| <= 2)
#'
#' Two additional orthogonal flags:
#'   Hyperplastic : high H_out AND out degree >= 3
#'                  identifies states that disperse flux broadly (plastic)
#'   Hub          : high betweenness AND both in and out degree >= 3
#'                  identifies obligate transitional crossroads
#'
#' Note: hub and hyperplastic are not mutually exclusive with the primary
#' classification — a transient state can be both a hub and hyperplastic.
classify_states <- function(metrics) {
  
  n          <- nrow(metrics)
  state_type <- rep("none", n)
  
  # Primary classification based on degree asymmetry and net flux direction
  # var_out - var_in reflects the imbalance between number of outgoing
  # and incoming connections, independently of weight magnitude
  is_sink     <- metrics$betweenness<boxplot(metrics$betweenness,plot=FALSE)$stats[3] &
    metrics$net_flow < 0
  is_source   <- metrics$betweenness<boxplot(metrics$betweenness,plot=FALSE)$stats[3] &
    metrics$net_flow > 0
  is_transient <- metrics$betweenness>=boxplot(metrics$betweenness,plot=FALSE)$stats[3]
  
  state_type[is_sink]      <- "Sink/attractor"
  state_type[is_source]    <- "Source/Progenitor"
  state_type[is_transient] <- "Transient"
  
  # Hub: high betweenness centrality AND connected to multiple states
  # in both directions — a genuine topological crossroads
  is_hub <- metrics$betweenness > quantile(metrics$betweenness, 0.25) &
    metrics$var_out >= 3 &
    metrics$var_in  >= 3
  
  # Hyperplastic: high outgoing entropy AND multiple outgoing connections
  # captures states that disperse flux broadly across many downstream fates
  is_hyperplastic <- metrics$H_out > boxplot(metrics$H_out,
                                             plot = FALSE)$stats[3] &
    metrics$var_out > 3
  
  metrics$state_type      <- state_type
  metrics$is_hub          <- ifelse(is_hub,          "Hub",     "standard")
  metrics$is_hyperplastic <- ifelse(is_hyperplastic, "plastic", "standard")
  
  metrics
}

# =============================================================================
# DATA LOADING AND GRAPH CONSTRUCTION
# =============================================================================

# Biological condition entropy per cluster
myentropy <- read.csv(
  "../cluster_mutcond_entropy_v2.csv")

# Transition probability matrix from partial optimal transport
# Partial OT allows mass to be created or destroyed, making it more
# appropriate than balanced OT when cluster sizes differ substantially
# and when some transitions should have zero probability
myp <- read.csv(
  "../transition_matrix_euclideanlinear.csv"
)[, -1]

for(i in c(1:ncol(myp))){
    myp[i,i]<-0
}

metrics_cedric <- read.csv(
  "../node_metric_results.csv"
)[, -1]

# ── Graph construction ────────────────────────────────────────────────────────
data           <- myp
colnames(data) <- rownames(data) <- paste("c", 1:12)

g <- graph_from_adjacency_matrix(
  as.matrix(data),
  mode     = "directed",   # asymmetric: P[i,j] != P[j,i] in general
  weighted = TRUE,         # edge weights = transition probabilities
  diag     = FALSE         # self-loops excluded: focus on inter-state transitions
)

# Edge cost for betweenness calculation: use inverse weight so that
# high-probability transitions correspond to short path lengths
E(g)$cost <- 1 / E(g)$weight

# Node-level strength (weighted degree)
V(g)$outstr <- strength(g, mode = "out", weights = E(g)$weight)
V(g)$instr  <- strength(g, mode = "in",  weights = E(g)$weight)

# ── Metric computation ────────────────────────────────────────────────────────
metrics <- data.frame(
  # Biological condition entropy: Shannon entropy of the distribution
  # of images across the 6 biological conditions within each cluster.
  # High entropy = cluster is populated by cells from many conditions
  # (biologically plastic); low entropy = condition-specific cluster.
  entropy = myentropy$entropy,
  
  # Weighted in and out strength: total incoming/outgoing transition
  # probability mass for each node
  instr  = strength(g, mode = "in",  weights = E(g)$weight),
  outstr = strength(g, mode = "out", weights = E(g)$weight),
  
  # Betweenness centrality: fraction of shortest paths passing through
  # each node, using inverse-weight costs so high-probability transitions
  # are treated as short distances
  betweenness = metrics_cedric$Betweenness.Centrality,
  
  # Outgoing and incoming transition entropy: Shannon entropy of the
  # probability distribution over outgoing/incoming edges.
  # H_out high = disperses to many targets equally (pluripotent)
  # H_in  high = receives from many sources equally (convergence point)
  H_out = metrics_cedric$Generalized.Entropy.Out,
  H_in  = metrics_cedric$Generalized.Entropy.In,     # 0 for nodes with no incoming edges
  
  # Mean entropy: average of H_in and H_out
  # High H_mu = both sends to and receives from diverse states = hub-like
  H_mu = (out_entropy(g) + in_entropy(g)) / 2,
  
  # Degree counts: number of non-zero outgoing/incoming connections
  # (unweighted; counts structural connectivity regardless of weight)
  var_out = apply(data > 0, 1, sum),
  var_in  = apply(data > 0, 2, sum),
  
  # PageRank: stationary distribution of a random walk with restart.
  # High PageRank = disproportionately visited = attractor-like
  page_rank = page_rank(g, directed = TRUE,
                        weights = E(g)$weight)$vector,
  
  # Net flow: outgoing minus incoming strength.
  # Positive = net sender (source); negative = net receiver (sink)
  net_flow = metrics_cedric$Net.Flow,
  
  # Send-receive ratio: relative balance on a ratio scale.
  # Offset by min(instr) to avoid division by zero for pure source nodes
  send_receive_ratio = (
    strength(g, mode = "out", weights = E(g)$weight) +
      min(strength(g, mode = "in",
                   weights = E(g)$weight)[
                     strength(g, mode = "in",
                              weights = E(g)$weight) > 0])
  ) / (
    strength(g, mode = "in",  weights = E(g)$weight) +
      min(strength(g, mode = "in",
                   weights = E(g)$weight)[
                     strength(g, mode = "in",
                              weights = E(g)$weight) > 0])
  )
)

# Apply state classification
metrics <- classify_states(metrics)

# =============================================================================
# VISUALISATION
# =============================================================================

# Edge width proportional to transition probability (amplified for visibility)
E(g)$width <- 200 * E(g)$weight^2.5

# Fix layout for reproducibility across all plots
set.seed(10)
layout_fixed  <- layout_with_fr(g)
g            <- prune_graph_for_viz(g, quantile_thresh = 0.6)


pdf("graph_analysis_partial_OT_pruned_CVC.pdf")

# ── Plot 1: Graph coloured by cluster identity ────────────────────────────────
par(mfrow = c(1, 1), mar = c(2, 2, 3, 2))
V(g)$color <- mycols
plot(
  g,
  layout          = layout_fixed,
  edge.arrow.size = 0.3,
  edge.width      = E(g)$width,
  edge.curved     = 0.4,
  edge.color      = "black",
  vertex.label.cex   = 0.9,
  vertex.label.color = "black"
)
title(main = "Morphological transition graph — cluster identity",
      cex.main = 0.9, col.main = "grey20")


# ── Plot 2: Graph coloured by state type ─────────────────────────────────────
color_map_state <- c(
  "Sink/attractor"    = "#C00000",   # red
  "Source/Progenitor" = "#2E75B6",   # blue
  "Transient"         = "#70AD47"    # green
)
V(g)$color <- color_map_state[metrics$state_type]
plot(
  g,
  layout          = layout_fixed,
  edge.arrow.size = 0.3,
  edge.width      = E(g)$width,
  edge.curved     = 0.3,
  edge.color      = "black",
  vertex.label.cex   = 0.9,
  vertex.label.color = "black"
)
legend("bottomleft",
       legend = names(color_map_state),
       fill   = color_map_state,
       title  = "State type",
       cex    = 0.8, bty = "n")
title(main = "State classification (sink / source / transient)",
      cex.main = 0.9, col.main = "black")

# ── Plot 3: Graph coloured by hub status ─────────────────────────────────────
color_map_hub <- c("standard" = "white", "Hub" = "#2E75B6")
V(g)$color    <- color_map_hub[metrics$is_hub]
plot(
  g,
  layout          = layout_fixed,
  edge.arrow.size = 0.3,
  edge.width      = E(g)$width,
  edge.curved     = 0.3,
  edge.color      = "black",
  vertex.label.cex   = 0.9,
  vertex.label.color = "black"
)
legend("bottomleft",
       legend = names(color_map_hub),
       fill   = color_map_hub,
       title  = "Hub status",
       cex    = 0.8, bty = "n")
title(main = "Hub states (high betweenness, connected in both directions)",
      cex.main = 0.9, col.main = "black")

# ── Plot 4: Graph coloured by hyperplastic status ────────────────────────────
color_map_plastic <- c("standard" = "white", "plastic" = "purple")
V(g)$color        <- color_map_plastic[metrics$is_hyperplastic]
plot(
  g,
  layout          = layout_fixed,
  edge.arrow.size = 0.3,
  edge.width      = E(g)$width,
  edge.curved     = 0.3,
  edge.color      = "black",
  vertex.label.cex   = 0.9,
  vertex.label.color = "black"
)
legend("bottomleft",
       legend = names(color_map_plastic),
       fill   = color_map_plastic,
       title  = "Hyperplastic",
       cex    = 0.8, bty = "n")
title(main = "Hyperplastic states (high H_out, broad outgoing connectivity)",
      cex.main = 0.9, col.main = "black")

# ── Plots 5-13: Continuous metric maps ───────────────────────────────────────
par(mfrow = c(2, 2))

metric_plots <- list(
  list(val = metrics$entropy,     name = "Biological condition entropy"),
  list(val = metrics$instr,       name = "Ingoing flow (weighted in-strength)"),
  list(val = metrics$outstr,      name = "Outgoing flow (weighted out-strength)"),
  list(val = metrics$H_in,        name = "Ingoing transition entropy"),
  list(val = metrics$H_out,       name = "Outgoing transition entropy"),
  list(val = metrics$net_flow,    name = "Net flow (out - in strength)"),
  list(val = metrics$betweenness, name = "Betweenness centrality"),
  list(val = metrics$var_out,     name = "Number of outgoing targets"),
  list(val = metrics$var_in,      name = "Number of incoming sources"),
  list(val = metrics$page_rank,      name = "Page rank")
  
)

for (mp in metric_plots) {
  PlotGraphColorsScale(valoi = mp$val, valname = mp$name, g = g, colramp = ramp2)
}

# ── Plot 14: Metric correlation panel ────────────────────────────────────────
par(mfrow = c(4, 4), mar = c(3, 3, 2, 2))
for (j in 2:ncol(metrics)) {
  if (is.numeric(metrics[, j])) plotmymetrics(i = j)
}

# ── Plots 15-17: Entropy by state classification ─────────────────────────────
boxplot(
  tapply(metrics[, 1],
         INDEX = factor(metrics$state_type,
                        levels = c("Transient", "Source/Progenitor",
                                   "Sink/attractor")),
         FUN = function(y) y),
  col    = c("#70AD47", "#2E75B6", "#C00000"),
  las    = 1,
  ylab   = "Biological condition entropy",
  main   = "Entropy by state type",
  frame  = FALSE
)

boxplot(
  tapply(metrics[, 1],
         INDEX = factor(metrics$is_hub,
                        levels = c("standard", "Hub")),
         FUN = function(y) y),
  col   = c("white", "#2E75B6"),
  las   = 1,
  ylab  = "Biological condition entropy",
  main  = "Entropy: hub vs standard",
  frame = FALSE
)

boxplot(
  tapply(metrics[, 1],
         INDEX = factor(metrics$is_hyperplastic,
                        levels = c("standard", "plastic")),
         FUN = function(y) y),
  col   = c("white", "purple"),
  las   = 1,
  ylab  = "Biological condition entropy",
  main  = "Entropy: hyperplastic vs standard",
  frame = FALSE
)

dev.off()