#不同工具之间比较
library(ggplot2)
library(dplyr)
data <- read.csv("/media/disk/project/0.paper/fig2data.csv")

data[1:3,]
dim(data)
# data <- filter(data,size == 1000)
data <- filter(data,size == 500)

dim(data)
dim = 13
data_norm <- apply(data[,6:dim],1,function(x){
    min = min(x)
    max = max(x)
    return((x-min)/(max-min))
})
data_norm <- as.data.frame(t(data_norm))

library(tidyr)
tmp <- gather(data_norm,key = "tools",value = "EPR",1:8)
tmp$tools <- factor(tmp$tools,levels = c("KGE.MAE","MAE","PIDC","GENIE3","GRNBOOST2","SCODE","PPCOR","SINCERITIES"))

ggplot(tmp,aes(x = tools,y = EPR,fill = tools))+
    geom_boxplot()+
    theme_bw()+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))+
    labs(title = "EPR",x = "Tools",y = "Normalized EPR")
ggsave("/media/disk/project/0.paper/fig2_rank_EPR.png",width = 6,height = 4)


data_norm$dataset <- data[1:21,"gt"]
data_norm$celltype <- data[1:21,"celltype"]
data_norm
library(tidyr)
# data_norm$merge <- paste(data_norm$dataset,data_norm$celltype,sep = "_")
tmp <- gather(data_norm,key = "tools",value = "EPR",1:8) 
tmp$tools <- factor(tmp$tools,levels = c("KGE.MAE","MAE","PIDC","GENIE3","GRNBOOST2","SCODE","PPCOR","SINCERITIES"))
ggplot(tmp,aes(x = tools,y = EPR))+
    geom_violin(width = 1.1)+
    geom_boxplot(width = 0.2)+
    geom_jitter(aes(color = celltype,shape =dataset ),size = 2)+ 
    # facet_wrap(~dataset,scales = "free_x")+
    theme_bw()+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))+
    labs(title = "EPR",x = "Tools",y = "Normalized EPR")
ggsave("/media/disk/project/0.paper/fig2_rank_EPR_point.png",width = 17,height = 10)






data_rank <- apply(data[,6:dim],1,function(x){
    rank = rank(-x)
    return(rank)
})
data_rank <- as.data.frame(t(data_rank))
tmp <- data.frame(tools = names(colSums(data_rank)),ranksum = colSums(data_rank))
tmp$tools <- factor(tmp$tools,levels = c("KGE.MAE","MAE","PIDC","GENIE3","GRNBOOST2","SCODE","PPCOR","SINCERITIES"))

ggplot(tmp,aes( x= tools,y=ranksum,fill = tools))+
    geom_bar(stat = "identity")+
    theme_bw()+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("/media/disk/project/0.paper/fig2_rank_sum.png",width = 6,height = 4)
data_rank$dataset <- paste(data$gt,data$celltype,sep = "_")
data_rank$gt <- data$gt
data_rank$celltype <- data$celltype
data_rank$gt <- factor(data_rank$gt,levels = c("STRING","Non-specific ChIP-Seq","Cell-type specific ChIP-Seq","log/gof"))
data_rank$gt <- factor(data_rank$gt,levels = c("STRING","Non-specific ChIP-Seq","Cell-type specific ChIP-Seq"))

ggplot(data_rank,aes(x= celltype,y = KGE.MAE))+
    geom_segment(aes(xend = celltype,yend = 1),color = "black",linetype="dashed")+
    geom_point(color = "red",size = 4)+
    # geom_segment(aes(xend = dataset,yend = 1),color = "black",linetype="dashed")+
    theme_bw()+
    theme(axis.text.x = element_text(size = 10, angle = 80, hjust = 1,face = "bold"))+
    theme(axis.text.y = element_text(size = 10,face = "bold"))+
    scale_y_continuous(limits = c(1,7),breaks = 1:7,labels = c("1","2","3","4","5","6","7"))+
    facet_wrap(~gt,scales = "free_x",nrow =1 )
    # facet_grid(.~gt,scales = "free_x",space = "free_y")

ggsave("/media/disk/project/0.paper/fig2_rank_KGE.png",width = 10,height = 6)


ggplot(data_rank,aes(x= celltype,y = MAE))+
    geom_segment(aes(xend = celltype,yend = 1),color = "black",linetype="dashed")+
    geom_point(color = "red",size = 4)+
    # geom_segment(aes(xend = dataset,yend = 1),color = "black",linetype="dashed")+
    theme_bw()+
    theme(axis.text.x = element_text(size = 10, angle = 80, hjust = 1,face = "bold"))+
    theme(axis.text.y = element_text(size = 10,face = "bold"))+
    scale_y_continuous(limits = c(1,7),breaks = 1:7,labels = c("1","2","3","4","5","6","7"))+
    facet_wrap(~gt,scales = "free_x",nrow =1 )
    # facet_grid(.~gt,scales = "free_x",space = "free_y")

ggsave("/media/disk/project/0.paper/fig2_rank_MAE.png",width = 10,height = 6)







#########比较不同KG(KEGG,TRRUST,cell type marker filter KG)以及linear layer的效果
data <- read.csv("/media/disk/project/0.paper/fig2.parameter_compare.csv")
data[1:3,1:3]
library(tidyr)
library(dplyr)

###TF+500
num = 500
tmp <- filter(data,size == num) %>% group_by(dataset,gt)%>% 
    mutate(value_normalized = (KGE.MAE - min(KGE.MAE, na.rm = TRUE)) / (max(KGE.MAE, na.rm = TRUE) - min(KGE.MAE, na.rm = TRUE))) %>% as.data.frame

tmp$gt <- factor(tmp$gt,levels = c("STRING","Non-specific ChIP-Seq","Cell-type specific ChIP-Seq"))
tmp$KG <- factor(tmp$KG,levels = c("CT_KEGG_L",'CT_KEGG',"TRRUST_L","KEGG","TRRUST"))
tmp$dataset <- factor(tmp$dataset,levels = c("mHSC-E",'mHSC-L',"MHSC-GM","mESC","mDC","hESC","hHep"))
library(ggplot2)
ggplot(tmp,aes( x= KG,y=value_normalized,fill = KG))+
    geom_bar(stat = "identity")+
    theme_bw()+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))+
    facet_wrap(~gt+dataset,scales = "free_x",ncol=7 )
ggsave("/media/disk/project/0.paper/tmp.png",width = 15,height = 10)


ggplot(tmp,aes(x = KG,y = value_normalized,fill = KG))+
    geom_boxplot()+
    theme_bw()+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))+
    labs(title = "KGE",x = "KG",y = "Normalized EPR")
ggsave("/media/disk/project/0.paper/tmp.png",width = 10,height = 7)

ggplot(tmp,aes(x = KG,y = value_normalized))+
    geom_violin(width = 1)+
    geom_boxplot(width = 0.2)+
    geom_jitter(aes(color = dataset,shape =gt ),size = 1.5)+ 
    # facet_wrap(~dataset,scales = "free_x")+
    theme_bw()+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))+
    labs(title = "KGE",x = "KG",y = "Normalized EPR")
ggsave("/media/disk/project/0.paper/tmp.png",width = 15,height = 7)


###TF+1000
num = 1000
tmp <- filter(data,size == num) 
ggplot(tmp,aes( x= KG,y=KGE.MAE,fill = KG))+
    geom_bar(stat = "identity")+
    theme_bw()+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))+
    facet_wrap(~gt+dataset,scales = "free_x",ncol=7 )
ggsave("/media/disk/project/0.paper/tmp.png",width = 15,height = 7)

ggplot(tmp,aes(x = KG,y = KGE.MAE,fill = KG))+
    geom_boxplot()+
    theme_bw()+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))+
    labs(title = "KGE",x = "KG",y = "Normalized EPR")
ggsave("/media/disk/project/0.paper/tmp.png",width = 5,height = 7)

ggplot(tmp,aes(x = KG,y = KGE.MAE))+
    geom_violin(width = 1)+
    geom_boxplot(width = 0.2)+
    geom_jitter(aes(color = dataset,shape =gt ),size = 1.5)+ 
    # facet_wrap(~dataset,scales = "free_x")+
    theme_bw()+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))+
    labs(title = "KGE",x = "KG",y = "Normalized EPR")
ggsave("/media/disk/project/0.paper/tmp.png",width = 5,height = 7)
