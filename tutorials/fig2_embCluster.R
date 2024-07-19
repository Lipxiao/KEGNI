## 根据EPR选择gene embedding 文件
library(ggplot2) 
library(dplyr)
library(clusterProfiler)
library(org.Hs.eg.db)
library(org.Mm.eg.db)
library(Seurat)
library(patchwork)
library(enrichplot)
library(stats)
library(factoextra)
library(fpc)

source('tutorials/utils.R')

# file = "./outputs/mHSC-E-ChIP-ExpressionData_2024-04-22-13-23-33-embedding.csv"
# file = "./outputs/mHSC-E-ChIP-ExpressionData_2024-05-10-17-18-21-embedding.csv"
# file = "./outputs/mHSC-E-ChIP-ExpressionData_2024-05-10-15-37-27-embedding.csv" 
# mHSC-E-ChIP-ExpressionData_2024-05-10-15-37-27
file = "script/outputs/mHSC-E_exp_2024-06-02-19-45-47-scg_embedding.csv"
dim = 30
kge <- reduction(file,dim)
clusterEval(kge,dim)
plot2D(kge)
# mHSC-E-ChIP-ExpressionData_2024-04-24-19-42-06-embedding.csv
# sh run_pretrain.sh mHSC-E 20 256 4 3 900 2 /media/disk/project/KnowledgeGraphEmbedding/data/KEGG/KEGG_mHSC-E.tsv
# mHSC-L-ChIP-ExpressionData_2024-04-24-19-44-22-embedding.csv
# sh run_pretrain.sh mHSC-L 20 256 4 3 900 2 /media/disk/project/KnowledgeGraphEmbedding/data/KEGG/KEGG_mHSC-L.tsv
# mHSC-GM-ChIP-ExpressionData_2024-04-24-19-45-45-embedding.csv
# sh run_pretrain.sh mHSC-GM 30 512 4 3 600 4 /media/disk/project/KnowledgeGraphEmbedding/data/KEGG/KEGG_mHSC-GM.tsv

file = "/media/disk/project/GRN/e2e_model/MAE/embedding/mHSC-E-ChIP-ExpressionData_20_256_4_3_350-embedding4explanation.csv"
dim = 30
mae <- reduction(file,dim)
clusterEval(mae,dim)
plot2D(mae)
# mHSC-E-ChIP-ExpressionData_20_256_4_3_350-embedding4explanation.csv
# sh run_MAE_sc.sh mHSC-E 20 256 4 3 350 6
#     mHSC-E-STRING  mHSC-E-NonSpe  mHSC-E-ChIP
# 0       0.153173       0.069474     0.584667

# mHSC-L-ChIP-ExpressionData_20_256_4_4_600-embedding4explanation.csv
# sh run_MAE_sc.sh mHSC-L 20 256 4 4 600 2
#     mHSC-L-STRING  mHSC-L-NonSpe  mHSC-L-ChIP
# 0       0.182482       0.075269     0.567985

# mHSC-GM-ChIP-ExpressionData_20_256_4_3_300-embedding4explanation.csv 
# sh run_MAE_sc.sh mHSC-GM 20 256 4 3 300 4
#     mHSC-GM-STRING  mHSC-GM-NonSpe  mHSC-GM-ChIP
# 0        0.136364        0.080754      0.607279


kge$RNA_snn_res.0.2
mae$RNA_snn_res.0.2
kge@active.ident <- kge$RNA_snn_res.0.2
mae@active.ident <- mae$RNA_snn_res.0.2
table(kge@active.ident,mae@active.ident)

###enrichment plot for GO and KEGG
### plot save 
organism = "mmu"
seurat_obj = kge
enrichPlot(seurat_obj,organism)


organism = "mmu"
seurat_obj = mae
enrichPlot(seurat_obj,organism)


# porphyrin metabolic process 和 heme metabolic process， tetrapyrrole metabolic process的基因在mae中可能也属于一个cluster，但是mae其他go term pvalue 更小
# cells <- case_conversion(colnames(subset(seurat_obj, idents = i)))
# print(length(cells))
# bitr <- bitr(cells, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = OrgDb)
# ego2 <- enrichGO(gene = bitr$ENTREZID, OrgDb = OrgDb, keyType = 'ENTREZID', ont = "BP", pAdjustMethod = "BH", pvalueCutoff  = 0.05)
# as.data.frame(ego2) %>% head
# a <- filter(ego2,ID == "GO:0033014")$geneID %>% strsplit(.,"/") %>% unlist %>% bitr(fromType = "ENTREZID", toType = "SYMBOL", OrgDb = OrgDb) 
# mae@meta.data[toupper(a$SYMBOL),"RNA_snn_res.0.2"]
# table(kge@active.ident)
# table(mae@active.ident)

##比较kge,mae的富集结果
OrgDb = org.Mm.eg.db
mae_list <- list()
for(i in as.character(levels(mae@active.ident))){
    print(i)
    mae_list <- c(mae_list,list(bitr(capitalize(tolower(colnames(subset(mae, idents = i)))), fromType = "SYMBOL", toType = "ENTREZID", OrgDb = OrgDb)$SYMBOL))
}
names(mae_list) <- paste("mae",as.character(levels(mae@active.ident)),sep = "_")

kge_list <- list()
for (i in as.character(levels(kge@active.ident))){
    print(i)
    kge_list <- c(kge_list,list(bitr(capitalize(tolower(colnames(subset(kge, idents = i)))), fromType = "SYMBOL", toType = "ENTREZID", OrgDb = OrgDb)$SYMBOL))
}
names(kge_list) <- paste("kge",as.character(levels(kge@active.ident)),sep = "_")

gcSample <- c(mae_list,kge_list)
# ck <- compareCluster(geneCluster = gcSample, fun = enrichKEGG)
ck <- compareCluster(geneCluster = gcSample, fun = 'enrichGO',keyType = "SYMBOL",OrgDb = OrgDb,ont = "BP",
                     pAdjustMethod = "BH",qvalueCutoff = 1e-3)
bp <- pairwise_termsim(ck)
bp2 <- simplify(bp, cutoff=0.9, by="p.adjust", select_fun=min)
# bp3 <- setReadable(bp2, OrgDb = OrgDb, keyType="ENTREZID")
# gofilter(bp2,level = 3)
edo <- group_by(bp2,Cluster)  %>% top_n(-10,qvalue) %>% arrange(Cluster,qvalue) 
dotplot(edo, showCategory = 15)+theme(axis.text.x = element_text(angle = 45,hjust = 1))
ggsave("./figures/fig3.go_dotplot.pdf",width = 12,height = 25)
ggsave("./figures/fig3.go_dotplot.eps",width = 12,height = 25)

# treeplot(edo)
# ggsave("./figures/tmp.pdf",width = 15,height = 15)
library(RColorBrewer)
emapplot(edo,showCategory = 12,)+
         scale_fill_manual(values = c(brewer.pal(8, "Blues")[3:6],brewer.pal(8, "Reds")[3:6],brewer.pal(8, "Greens")[4:6],brewer.pal(8, "Purples")[4:6]))
ggsave("./figures/fig3.emapplot.pdf",width = 10,height = 10)
ggsave("./figures/fig3.emapplot.eps",width = 10,height = 10)

###自定义画图
bp3 <- as.data.frame(edo)
bp3$group <- ifelse(bp3$Cluster %in% names(mae_list), "mae", "kge")

bp3$Cluster <- as.character(bp3$Cluster)
library(tidyr)
ck_show <-  as.data.frame(group_by(filter(bp3,qvalue < 0.05),Cluster) %>% top_n(-12,qvalue) %>% arrange(Cluster,qvalue,Description) )
ck_show$Description <- factor(ck_show$Description,levels = unique(ck_show$Description))
ggplot(ck_show, aes(x = Cluster, y = Description, fill = -log10(qvalue))) +
    geom_point(aes(size = sapply(GeneRatio,function(x) eval(parse(text = x)))), shape = 21) +
    # geom_point() +
    scale_size(range = c(3,8)) +
    scale_fill_gradient(low = "blue", high = "red") +
    # theme(legend.position = "none") +
    facet_wrap(~group,scales = "free_x",nrow = 1) +
    theme(text = element_text(size=14,face = "bold"),
          axis.line = element_line(colour = "black",linewidth = 0.5),
          axis.text.x = element_text(angle = 45,hjust = 1))+
    guides(size = guide_legend(title = "GeneRatio"))
ggsave("./figures/fig3.tmp.pdf",width = 10,height = 10)



sankey_data <- as.data.frame(table(kge@active.ident,mae@active.ident))
colnames(sankey_data) <- c("source","target","value")
library(ggalluvial)
ggplot(data = sankey_data,
    aes(axis1 = source, axis2 = target,y = value)) +
    geom_alluvium(aes(fill = source)) +
    geom_stratum() +
    geom_text(stat = "stratum", aes(label = after_stat(stratum))) +
    # scale_fill_manual(values = c("#66c2a5", "#fc8d62", "#8da0cb")) +
    theme(legend.position = "none")+
    theme_minimal() 
ggsave("./figures/fig3.sankey.png",width = 4,height = 3)
ggsave("./figures/fig3.sankey.pdf",width = 4,height = )



## 按照上面cluster中的gene数目随机采样生成随机cluster
##kge
set.seed(42)
random_gene <- capitalize(tolower(sample(colnames(kge_epr))))
random_cluster <- rep(0:3,table(kge_epr@active.ident)) 
names(random_cluster) <- random_gene
table(random_cluster)
OrgDb = org.Mm.eg.db
organism='mmu'
kge_random_list <- list()
for(i in (unique(random_cluster))){
    print(i)
    kge_random_list <- c(kge_random_list,list(bitr(capitalize(tolower(names(random_cluster[random_cluster==i]))), fromType = "SYMBOL", toType = "ENTREZID", OrgDb = OrgDb)$SYMBOL))
}
names(kge_random_list) <- paste("kge_random",unique(random_cluster),sep = "_")
##富集分析
for (i in unique(random_cluster)){
    cells <- names(random_cluster[random_cluster==i])
    print(length(cells))
    bitr <- bitr(cells, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = OrgDb)
    kk <- enrichKEGG(bitr$ENTREZID, organism = organism,  pvalueCutoff = 0.05)
    if (dim(kk)[1] != 0){
        print(paste("kegg",i,sep = " "))
        p1 <- dotplot(kk, showCategory = 15)+ggtitle(paste("KEGG","cluster",i,sep = "_"))
        # p2 <- goplot(kk, showCategory = 10)+ggtitle(paste("KEGG","cluster",i,sep = "_"))
        p1
        ggsave(paste("./figures/","kegg",i,".png"),width = 8,height = 10)
    }
    ego2 <- enrichGO(gene = bitr$ENTREZID, OrgDb = OrgDb, keyType = 'ENTREZID', ont = "BP", pAdjustMethod = "BH", pvalueCutoff  = 0.05)
    if (dim(ego2)[1] != 0){
        print(paste("go",i,sep = " "))
        p1 <- dotplot(ego2, showCategory = 15)+ggtitle(paste("GO","cluster",i,sep = "_"))
        p2 <- goplot(ego2, showCategory = 10)+ggtitle(paste("GO","cluster",i,sep = "_"))
        p1+p2
        ggsave(paste("./figures/","go",i,".png"),width = 20,height = 10)
    }
}   
###mae
set.seed(42)
random_gene <- capitalize(tolower(sample(colnames(mae_epr))))
random_cluster <- rep(levels(mae_epr@active.ident),table(mae_epr@active.ident)) 
names(random_cluster) <- random_gene
table(random_cluster)
OrgDb = org.Mm.eg.db
organism='mmu'
mae_random_list <- list()
for(i in (unique(random_cluster))){
    print(i)
    mae_random_list <- c(mae_random_list,list(bitr(capitalize(tolower(names(random_cluster[random_cluster==i]))), fromType = "SYMBOL", toType = "ENTREZID", OrgDb = OrgDb)$SYMBOL))
}
names(mae_random_list) <- paste("mae_random",unique(random_cluster),sep = "_")
##富集分析
for (i in unique(random_cluster)){
    cells <- names(random_cluster[random_cluster==i])
    print(length(cells))
    bitr <- bitr(cells, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = OrgDb)
    kk <- enrichKEGG(bitr$ENTREZID, organism = organism,  pvalueCutoff = 0.05)
    if (dim(kk)[1] != 0){
        print(paste("kegg",i,sep = " "))
        p1 <- dotplot(kk, showCategory = 15)+ggtitle(paste("KEGG","cluster",i,sep = "_"))
        # p2 <- goplot(kk, showCategory = 10)+ggtitle(paste("KEGG","cluster",i,sep = "_"))
        p1
        ggsave(paste("./figures/","kegg",i,".png"),width = 8,height = 10)
    }
    ego2 <- enrichGO(gene = bitr$ENTREZID, OrgDb = OrgDb, keyType = 'ENTREZID', ont = "BP", pAdjustMethod = "BH", pvalueCutoff  = 0.05)
    if (dim(ego2)[1] != 0){
        print(paste("go",i,sep = " "))
        p1 <- dotplot(ego2, showCategory = 15)+ggtitle(paste("GO","cluster",i,sep = "_"))
        p2 <- goplot(ego2, showCategory = 10)+ggtitle(paste("GO","cluster",i,sep = "_"))
        p1+p2
        ggsave(paste("./figures/","go",i,".png"),width = 20,height = 10)
    }
}   



OrgDb = org.Mm.eg.db

gcSample <- c(mae_list,kge_list,mae_random_list,kge_random_list)
ck <- compareCluster(geneCluster = gcSample, fun = 'enrichGO',keyType = "SYMBOL",OrgDb = OrgDb,ont = "BP",
                     pAdjustMethod = "BH",qvalueCutoff = 1e-3)
bp <- pairwise_termsim(ck)
bp2 <- simplify(bp, cutoff=0.9, by="p.adjust", select_fun=min)
# bp3 <- setReadable(bp2, OrgDb = OrgDb, keyType="ENTREZID")
# gofilter(bp2,level = 3)
edo <- group_by(bp2,Cluster)  %>% top_n(-10,qvalue) #%>% arrange(Cluster,qvalue) 
# treeplot(edo)
dotplot(edo,showCategory = 10)+theme(axis.text.x = element_text(angle = 45,hjust = 1))
# ggsave("./figures/fig3.go_dotplot_all.png",width = 12,height = 25)
# ggsave("./figures/fig3.go_dotplot_all.pdf",width = 12,height = 25)

# treeplot(edo)
# ggsave("./figures/tmp.pdf",width = 15,height = 15)
library(RColorBrewer)
emapplot(edo,showCategory = 10,)+
        #  scale_fill_manual(values = c(brewer.pal(12, "Paired"),"gray","black"))

        #  scale_fill_manual(values =c("#C6DBEF","#FCBBA1","#A1D99B","#BCBDDC","#9ECAE1","#FC9272","#74C476","#9E9AC8","#6BAED6","#FB6A4A","#41AB5D","#807DBA","#4292C6","#EF3B2C"))
         scale_fill_manual(values = c(brewer.pal(8, "Blues")[3:6],brewer.pal(8, "Reds")[3:6],brewer.pal(8, "Greens")[5:6],brewer.pal(8, "Purples")[5:6]))
ggsave("./figures/fig3.go_emapplot_all.pdf",width = 10,height = 10)

bp3 <- as.data.frame(edo)
bp3$group <- ifelse(bp3$Cluster %in% names(mae_list), "mae", 
                    ifelse(bp3$Cluster %in% names(mae_random_list), "mae_random", 
                    ifelse(bp3$Cluster %in% names(kge_list), "kge", "kge_random")))
bp3$group <- factor(bp3$group,levels = c("kge","mae","kge_random","mae_random"))
bp3$Cluster <- as.character(bp3$Cluster)
library(tidyr)
ck_show <-  as.data.frame(group_by(filter(bp3,qvalue < 0.05),Cluster) %>% top_n(-10,qvalue) %>% arrange(Cluster,qvalue,Description) )
ck_show$Description <- factor(ck_show$Description,levels = unique(ck_show$Description))
ggplot(ck_show, aes(x = Cluster, y = Description, fill = -log10(qvalue))) +
    geom_point(aes(size = sapply(GeneRatio,function(x) eval(parse(text = x)))), shape = 21) +
    # geom_point() +
    scale_size(range = c(3,8)) +
    scale_fill_gradient(low = "blue", high = "red") +
    # theme(legend.position = "none") +
    facet_wrap(~group,scales = "free_x",nrow = 1) +
    theme(text = element_text(size=14,face = "bold"),
          axis.line = element_line(colour = "black",linewidth = 0.5),
          axis.text.x = element_text(angle = 45,hjust = 1))+
    guides(size = guide_legend(title = "GeneRatio"))
          
ggsave("./figures/fig3.go_dotplot_all.png",width = 15,height = 15)
ggsave("./figures/fig3.go_dotplot_all.pdf",width = 15,height = 15)


ck_show <-  as.data.frame(group_by(filter(bp3,qvalue < 0.01),Cluster) %>% top_n(-10,qvalue) %>% arrange(Cluster,qvalue,Description) )
ck_show <- ck_show[ck_show$Description !="protein localization to chromatin",] 
library(GOSemSim)
mmGO <- godata('org.Mm.eg.db', ont="BP")
ego.sim <- mgoSim(unique(ck_show$ID), unique(ck_show$ID), semData=mmGO, measure="Wang", combine=NULL)
ego.sim[1:3, 1:3]
# ck_show[!(duplicated(ck_show$ID)),"Description"] 

#用GO term作为行名、列名，便于查看和画图
rownames(ego.sim) <- ck_show[!(duplicated(ck_show$ID)),"Description"] 
colnames(ego.sim) <- ck_show[!(duplicated(ck_show$ID)),"Description"] 
ego.sim[1:3, 1:3]

distance_matrix_from_similarity <- as.dist(1 - ego.sim)
hc <- hclust(distance_matrix_from_similarity, method = "average")
nCluster = 4
clus <- stats::cutree(hc, nCluster)
cluster_df <- data.frame(label = names(clus), cluster = as.factor(clus))

library(ape)
library(aplot)
library(ggtree)
p2 <- ggtree(as.phylo(hc), branch.length='none')+
    # geom_tiplab()+
    theme_tree2()+
    xlim(NA,28) 
p2 <- p2 %<+% cluster_df + #geom_tippoint(aes(color = cluster),size = 4) +
    geom_label(aes(label = label,fill = cluster,alpha = 0.5),hjust = 0, label.size = 0) +
    theme(legend.position = "left") 
p2

library(aplot)

p1 <- ggplot(ck_show, aes(x = Cluster, y = Description, fill = -log10(qvalue))) +
    geom_point(aes(size = sapply(GeneRatio,function(x) eval(parse(text = x)))), shape = 21) +
    # geom_point() +
    scale_size(range = c(3,8)) +
    scale_fill_gradient(low = "blue", high = "red") +
    # theme(legend.position = "none") +
    facet_wrap(~group,scales = "free_x",nrow = 1) +
    theme(text = element_text(size=14,face = "bold"),
          axis.line = element_line(colour = "black",linewidth = 0.5),
          axis.text.y = element_blank(),
          axis.text.x = element_text(angle = 45,hjust = 1))+
    guides(size = guide_legend(title = "GeneRatio"))+
    labs(x=NULL,y=NULL)
p1

p1%>%insert_left(p2,width = 1)
ggsave("figures/fig3.go_dotplot_all.pdf",width = 20,height = 15)
ggsave("figures/fig3.go_dotplot_all.png",width = 15,height = 15)

library(cluster)
library(mclust)
library(fpc)
type = 'mae_random'
a <- ck_show[ck_show$group == type,]$Cluster %>% strsplit(paste(type,"_",sep = "")) %>% sapply(function(x) x[2]) %>% as.numeric
b <- c()
for (i in ck_show[ck_show$group ==type,]$Description){
    b <-c(b,filter(cluster_df,cluster_df$label == i)$cluster)
}
adjustedRandIndex(a,b)
library(mclustcomp)
mclustcomp(a, b, types = "all", tversky.param = list())

### 随机选取两个mHSC-E gene embedding 进行聚类
# mHSC-E-ChIP-ExpressionData_2024-04-22-14-26-45-embedding.csv
# mHSC-E-ChIP-ExpressionData_2024-04-22-13-49-43-embedding.csv
# mHSC-E-ChIP-ExpressionData_30_128_4_3_400-embedding.csv
# mHSC-E-ChIP-ExpressionData_30_256_4_3_250-embedding.csv

file = "./outputs/mHSC-E-ChIP-ExpressionData_2024-04-22-14-26-45-embedding.csv"
dim = 30
kge <- reduction(file,dim)
plot2D(kge)#到figures文件夹查看tSNE和UMAP图

kge$RNA_snn_res.0.2
kge@active.ident <- kge$RNA_snn_res.0.2

organism = "mmu"
seurat_obj = kge
enrichPlot(seurat_obj,organism)


file = "./outputs/mHSC-E-ChIP-ExpressionData_2024-04-22-13-49-43-embedding.csv"
dim = 30
kge <- reduction(file,dim)
plot2D(kge)#到figures文件夹查看tSNE和UMAP图

kge$RNA_snn_res.0.2
kge@active.ident <- kge$RNA_snn_res.0.2
organism = "mmu"
seurat_obj = kge
enrichPlot(seurat_obj,organism)




file = "./MAE/embedding/mHSC-E-ChIP-ExpressionData_30_128_4_3_400-embedding.csv"
dim = 30
mae <- reduction(file,dim)
plot2D(mae)

mae$RNA_snn_res.0.2
mae@active.ident <- mae$RNA_snn_res.0.2
organism = "mmu"
seurat_obj = mae
enrichPlot(seurat_obj,organism)



file = "./MAE/embedding/mHSC-E-ChIP-ExpressionData_30_256_4_3_250-embedding.csv"
dim = 30
mae <- reduction(file,dim)
plot2D(mae)

mae$RNA_snn_res.0.1
mae@active.ident <- mae$RNA_snn_res.0.1
organism = "mmu"
seurat_obj = mae
enrichPlot(seurat_obj,organism)


gene <-  names(kge@active.ident)
upset <- data.frame("gene" = gene,"kge_epr" = unfactor(kge_epr@active.ident),
           "kge_random1" = unfactor(kge_random1@active.ident),
           "kge_random2" = unfactor(kge_random2@active.ident),
           "mae_epr" = unfactor(mae_epr@active.ident),
           "mae_random1" = unfactor(mae_random1@active.ident),
           "mae_random2" = unfactor(mae_random2@active.ident))

upset$kge <- paste(upset$kge_epr,upset$kge_random1,upset$kge_random2,sep = "_")
unique(upset$kge)
sankey_data <- as.data.frame(table(upset$kge))
sankey_data$Var1 <- as.character(sankey_data$Var1)
sankey_data[,3:5] <- as.data.frame(strsplit(sankey_data$Var1,split = "_") %>% unlist %>% matrix(.,ncol=3,,byrow = T))
colnames(sankey_data) <- c("type","Freq","kge_epr","kge_random1","kge_random2")
library(ggalluvial)
ggplot(data = sankey_data,
    aes(axis1 = kge_random1, axis2 = kge_epr, axis3 = kge_random2,
        y = Freq)) +
    scale_x_discrete(limits = c("kge_random1", "kge_epr", "kge_random2"), expand = c(.2, .05)) +
    xlab("Demographic") +
    geom_alluvium(aes(fill = kge_epr)) +
    geom_stratum() +
    geom_text(stat = "stratum", aes(label = after_stat(stratum))) +
    # scale_fill_manual(values = c("#66c2a5", "#fc8d62", "#8da0cb")) +
    theme(legend.position = "none")+
    theme_minimal() 
ggsave("./figures/kge_sankey.png",width = 6,height = 4)

upset$mae <- paste(upset$mae_epr,upset$mae_random1,upset$mae_random2,sep = "_")
unique(upset$mae)
sankey_data <- as.data.frame(table(upset$mae))
sankey_data$Var1 <- as.character(sankey_data$Var1)
sankey_data[,3:5] <- as.data.frame(strsplit(sankey_data$Var1,split = "_") %>% unlist %>% matrix(.,ncol=3,,byrow = T))
colnames(sankey_data) <- c("type","Freq","mae_epr","mae_random1","mae_random2")
library(ggalluvial)
ggplot(data = sankey_data,
    aes(axis1 = mae_random1, axis2 = mae_epr, axis3 = mae_random2,
        y = Freq)) +
    scale_x_discrete(limits = c("mae_random1", "mae_epr", "mae_random2"), expand = c(.2, .05)) +
    xlab("Demographic") +
    geom_alluvium(aes(fill = mae_epr)) +
    geom_stratum() +
    geom_text(stat = "stratum", aes(label = after_stat(stratum))) +
    # scale_fill_manual(values = c("#66c2a5", "#fc8d62", "#8da0cb")) +
    theme(legend.position = "none")+
    theme_minimal() 
ggsave("./figures/mae_sankey.png",width = 6,height = 4)







###mHSC-L
file = "./outputs/mHSC-L-ChIP-ExpressionData_2024-04-24-19-44-22-embedding.csv"
dim = 30
kge <- reduction(file,dim)
clusterEval(kge,dim)
plot2D(kge)

file = "./MAE/embedding/mHSC-L-ChIP-ExpressionData_20_256_4_4_600-embedding4explanation.csv"
dim = 30
mae <- reduction(file,dim)
clusterEval(mae,dim)
plot2D(mae)



kge$RNA_snn_res.0.2
mae$RNA_snn_res.0.2
kge@active.ident <- kge$RNA_snn_res.0.2
mae@active.ident <- mae$RNA_snn_res.0.2
table(kge@active.ident,mae@active.ident)


organism = "mmu"
seurat_obj = kge
enrichPlot(seurat_obj,organism)

organism = "mmu"
seurat_obj = mae
enrichPlot(seurat_obj,organism)




###mHSC-GM
# file = "./outputs/mHSC-GM-ChIP-ExpressionData_2024-04-24-19-45-45-embedding.csv"
file = "./outputs/mHSC-GM-ChIP-ExpressionData_2024-05-10-13-46-26-embedding.csv"
dim = 30
kge <- reduction(file,dim)
clusterEval(kge,dim)
plot2D(kge)

file = "./MAE/embedding/mHSC-GM-ChIP-ExpressionData_20_256_4_3_300-embedding4explanation.csv"
dim = 30
mae <- reduction(file,dim)
clusterEval(mae,dim)
plot2D(mae)


kge$RNA_snn_res.0.1
mae$RNA_snn_res.0.3
kge@active.ident <- kge$RNA_snn_res.0.1
mae@active.ident <- mae$RNA_snn_res.0.3
table(kge@active.ident,mae@active.ident)


organism = "mmu"
seurat_obj = kge
enrichPlot(seurat_obj,organism)

organism = "mmu"
seurat_obj = mae
enrichPlot(seurat_obj,organism)
