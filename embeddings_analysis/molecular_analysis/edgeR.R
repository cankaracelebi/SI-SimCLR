library(limma)
library(edgeR)


results_path <- "/idiap/temp/group/genomics/AstroBF_fastq/RN22346/230106_A01366_0335_BHJJHNDSX5/genes_preprocessing/DGE"

# Load the data
count_matrix <- as.matrix(read.csv(file = paste(results_path, "/df_bulk_for_dge.csv", sep=""), row.names = 1))
design <- as.data.frame(read.csv(file = paste(results_path, "/design_for_dge_grouped.csv", sep="")))


design$X <- design$X +1
count_matrix <- count_matrix[, design$X]

# ---- 2. Define factors ----
design$Condition <- factor(gsub(".*_", "", design$Treatment))  # basal, A1, etc
design$Condition <- factor(gsub(" ", "", design$Condition))    # cleanup spaces if any
design$CellType  <- factor(gsub("_.*", "", design$Treatment))  # ctrl vs vcp
design$Condition <- factor(design$Condition,
                           levels = c("basal", "A1low", "A1high"))
design$CellType  <- factor(design$CellType,
                           levels = c("ctrl", "vcp"))
# ---- 3. Build DGEList ----
dge <- DGEList(counts = count_matrix, samples = design, group = design$CellType)

# Filter low counts
keep <- filterByExpr(dge, group=design$CellType)
dge <- dge[keep,, keep.lib.sizes=FALSE]

# Normalize
dge <- calcNormFactors(dge)

# ---- 4. Design matrix ----
# ~ Condition + CellType  means:
# - Control for Condition (treatment effect)
# - Test CellType (ctrl vs vcp) while regressing out treatment
design_matrix <- model.matrix(~ Condition + CellType, data=design)
colnames(design_matrix)
# (Intercept) ConditionA1high ConditionA1low CellTypevcp

# ---- 5. Fit GLM ----
dge <- estimateDisp(dge, design_matrix)
fit <- glmQLFit(dge, design_matrix)

# ---- 6. Tests ----
# 6a. CellType effect (ctrl vs vcp), adjusting for treatment
res_ctrl_vs_vcp <- glmQLFTest(fit, coef="CellTypevcp")
write.csv(topTags(res_ctrl_vs_vcp, n=Inf), 
          file="dge_ctrl_vs_vcp_adjusted_for_treatment.csv")


# 6b. Treatment effects, adjusting for CellType
res_A1high_vs_basal <- glmQLFTest(fit, coef="ConditionA1high")
res_A1low_vs_basal  <- glmQLFTest(fit, coef="ConditionA1low")

write.csv(topTags(res_A1high_vs_basal, n=Inf), 
          file="dge_A1high_vs_basal_adjusted_for_celltype.csv")
write.csv(topTags(res_A1low_vs_basal, n=Inf), 
          file="dge_A1low_vs_basal_adjusted_for_celltype.csv")

# ---- 7. (Optional) Interaction model ----
# If you suspect ctrl vs vcp *respond differently* to treatment
design_matrix_int <- model.matrix(~ CellType * Condition, data=design)
colnames(design_matrix_int)
dge <- estimateDisp(dge, design_matrix_int)
fit_int <- glmQLFit(dge, design_matrix_int)

# Example: Is the effect of A1 high different in vcp vs ctrl?
res_interaction_high <- glmQLFTest(fit_int, coef="CellTypevcp:ConditionA1high")
write.csv(topTags(res_interaction_high, n=Inf), 
          file="dge_interaction_CellType_vcp_A1high.csv")


extract_gene_lists <- function(result, prefix, fdr=0.1, pval=0.05) {
  res_table <- topTags(result, n=Inf)$table
  sig_genes <- subset(res_table, FDR < fdr & PValue < pval)
  
  up   <- rownames(sig_genes[sig_genes$logFC > 0, ])
  down <- rownames(sig_genes[sig_genes$logFC < 0, ])
  
  write.table(up,   file=paste0("genes_up_", prefix, ".txt"),
              quote=FALSE, row.names=FALSE, col.names=FALSE)
  write.table(down, file=paste0("genes_down_", prefix, ".txt"),
              quote=FALSE, row.names=FALSE, col.names=FALSE)
}

# Usage
extract_gene_lists(res_ctrl_vs_vcp, "ctrl_vs_vcp")
extract_gene_lists(res_A1high_vs_basal, "A1high_vs_basal")
extract_gene_lists(res_A1low_vs_basal, "A1low_vs_basal")