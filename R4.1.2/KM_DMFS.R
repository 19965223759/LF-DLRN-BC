library("survival")
library("survminer")
library(ggthemes)



raw_data<-read.csv("./nomogram_cox_clinical.csv",stringsAsFactors = FALSE,fileEncoding='GBK')
range(raw_data$DMFS)
str(raw_data)
fit <- survfit(Surv(DMFS, metastasis) ~ group, data = raw_data)
#plot(fit, pval = T, conf.int = TRUE,break.time.by=300,xlim = c(0,1800))
jpeg("DMFS.jpeg",width=200,height = 200,units="mm",res=1000)
B<-ggsurvplot(fit ,ggtheme = theme_few(),
           pval = T,
           legend.title="Strata",
           legend.labs=c("Low metastasis risk group", "High metastasis risk group"),
           censor=F,
           risk.table=F,
           xlab="Time(days)",
           ylab="Distant Metastasis-Free survival",
           xlim = c(0,1800),
           pval.coord = c(1200,0.75),
           conf.int = T,
           lwd=2,
           break.time.by=300)
c <- B$plot+ 
  theme(legend.background = element_rect(fill="white",colour = "black",size=0.3),
        legend.position = c(0,0),legend.justification = c(0,0),
        legend.key.size = unit(20, "pt")
        )
c
dev.off()
