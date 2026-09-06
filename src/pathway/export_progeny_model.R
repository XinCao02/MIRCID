args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- normalizePath(sub("^--file=", "", file_arg[[1]]))
repo <- normalizePath(file.path(dirname(script_path), "..", ".."))
model_path <- file.path(repo, "src", "legacy_snapshot", "progeny_official", "model_human_full.rda")
load(model_path)

target_pathways <- c("EGFR", "Hypoxia", "JAK-STAT", "MAPK", "NFkB", "PI3K", "TGFb", "TNFa", "Trail", "VEGF", "p53")
sub <- model_human_full[as.character(model_human_full$pathway) %in% target_pathways, ]
sub$pathway <- as.character(sub$pathway)
sub$gene <- as.character(sub$gene)
sub <- do.call(rbind, lapply(split(sub, sub$pathway), function(x) x[order(x$p.value), ][seq_len(100), ]))
sub <- sub[order(match(sub$pathway, target_pathways), sub$p.value, sub$gene), ]
if (nrow(sub) != 1100 || any(table(sub$pathway) != 100)) stop("Top-100 model extraction failed")
write.csv(sub, file.path(repo, "data", "processed", "progeny_top100_long.csv"), row.names = FALSE)

genes <- sort(unique(sub$gene))
mat <- matrix(0, nrow = length(genes), ncol = length(target_pathways), dimnames = list(genes, target_pathways))
for (i in seq_len(nrow(sub))) mat[sub$gene[[i]], sub$pathway[[i]]] <- sub$weight[[i]]
write.csv(data.frame(gene = rownames(mat), mat, check.names = FALSE), file.path(repo, "data", "processed", "progeny_top100_matrix.csv"), row.names = FALSE)
cat(sprintf("PASS: exported %d nonzero official PROGENy weights (%d genes x %d pathways)\n", nrow(sub), nrow(mat), ncol(mat)))
