reduction <- function(emb_file,dim){
    print(emb_file)
    emb <- read.csv(emb_file,row.names = 1)
    print(emb[1:3,1:3])
    pbmc <- CreateSeuratObject(counts = t(emb),  min.cells = 0, min.features = 0)
    print(pbmc)
    # pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = min(3000, nrow(pbmc)))
    all.genes <- rownames(pbmc)
    pbmc <- ScaleData(pbmc, features = all.genes)

    # pbmc <- RunPCA(pbmc,features = VariableFeatures(object = pbmc),verbose = FALSE)
    pbmc <- RunPCA(pbmc,features = all.genes,verbose = FALSE)

    dim =dim
    pbmc <- FindNeighbors(pbmc, dims = 1:dim)
    for (i in seq(0.1,0.5,0.1)){
        pbmc <- FindClusters(pbmc, resolution = i,algorithm = 1)
    }
    for (k in seq(3,7,1)){
        kmeans_result <- kmeans(pbmc@reductions$pca@cell.embeddings[,1:dim], centers = k)
        var <- paste("kmeans_result",k,sep = "_")
        pbmc[[var]] <- kmeans_result$cluster
    }

    return(pbmc)
}


plot2D <- function(seurat_obj){
    # clusters = grep("RNA_snn_res",colnames(seurat_obj@meta.data),value = T)
    clusters <- c(grep("RNA_snn_res",colnames(seurat_obj@meta.data),value =T),grep("kmeans",colnames(seurat_obj@meta.data),value =T))
    for (cluster in clusters){
        # resolution = strsplit(cluster,"RNA_snn_res.")[[1]][2]
        seurat_obj@active.ident <- factor(seurat_obj@meta.data[,cluster])
        seurat_obj <- RunUMAP(seurat_obj, dims = 1:dim)
        DimPlot(seurat_obj, reduction = "umap",label = T)+ ggtitle(cluster)
        ggsave(paste("./figures/","umap_",cluster,".png",sep = ""),width = 3.75,height = 3)
        ggsave(paste("./figures/","umap_",cluster,".pdf",sep = ""),width = 3.75,height = 3)


        seurat_obj <- RunTSNE(seurat_obj, dims = 1:dim,check_duplicates = F)
        DimPlot(seurat_obj, reduction = "tsne",label = T)+ ggtitle(cluster)
        ggsave(paste("./figures/","tsne_",cluster,".png",sep = ""),width = 3.75,height = 3)
        ggsave(paste("./figures/","tsne_",cluster,".pdf",sep = ""),width = 3.75,height = 3)
    }

}


enrichPlot <- function(seurat_obj,organism){
    if (organism == "hsa"){
        OrgDb = org.Hs.eg.db
        case_conversion <- function(x) {toupper(x)}
    }else if (organism == "mmu") {
        OrgDb = org.Mm.eg.db
        library(Hmisc)
        case_conversion <- function(x) {capitalize(tolower(x))}

    }
    for (i in as.character(levels(seurat_obj@active.ident))){
        cells <- case_conversion(colnames(subset(seurat_obj, idents = i)))
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
            p1 <- dotplot(ego2, showCategory = 20)+ggtitle(paste("GO","cluster",i,sep = "_"))
            p2 <- goplot(ego2, showCategory = 15)+ggtitle(paste("GO","cluster",i,sep = "_"))
            p1+p2
            ggsave(paste("./figures/","go",i,".png"),width = 20,height = 10)
        }
    }   
    print("figures saved in /media/disk/project/GRN/KEGNI/figures")
}


clusterEval <- function(object,dim){
    # silhouette_scores <- silhouette(pbmc$kmeans_result_2, dist(pbmc@reductions$pca@cell.embeddings[,1:dim]))
    # fviz_cluster(kmeans_result, data = pca_data, geom = "point", stand = FALSE,palette = "jco", ggtheme = theme_minimal())
    clusters <- c(grep("RNA_snn_res",colnames(object@meta.data),value =T),grep("kmeans",colnames(object@meta.data),value =T))
    for (cluster in clusters){
        tmp <- cluster.stats(dist(object@reductions$pca@cell.embeddings[,1:dim]),as.numeric(object@meta.data[,cluster]))
        cat(cluster, "\nCHI:", tmp$ch, " DBI:", tmp$dunn, " SC:", tmp$avg.silwidth, "\n")
        #CHI指标由分离度与紧密度的比值得到,CHI越大越好
        #DBI越小越好
        #SC越大越好
    }
}
