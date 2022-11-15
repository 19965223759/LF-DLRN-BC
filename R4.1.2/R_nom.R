library(rms)
library(nomogramFormula)
raw_data<-read.csv("./nomogram_cox_clinical.csv",stringsAsFactors = FALSE,fileEncoding='GBK')
str(raw_data)
lrm_fit
dd <- datadist(raw_data)
options(datadist="dd")
str(raw_data)
log_multi<- as.formula(paste0('metastasis~', paste0(c('ki67','Histologic.type','PMI.T4','late_fusion_probability'),collapse='+')))
res_cph <- lrm(log_multi,data=raw_data)
surv3<-function(x)1/(1+exp(-x))
res_nomo <- nomogram(res_cph ,lp=F,fun=surv3, # 10年
                     fun.at=c(0.01,.1,.5,.7,.8),
                     funlabel='Risk of BC Metastasis')
results <- formula_rd(nomogram = res_nomo)
raw_data$points <- points_cal(formula = results$formula, rd=raw_data)
jpeg("nomogram-3.jpeg",width=270,height = 160,  units="mm",res=1000)
plot(res_nomo)
cindex.orig<-rcorrcens( metastasis~ predict(res_cph), data =raw_data )

cindex.orig

c<-rcorrcens(metastasis~ predict(res_cph), data =raw_data )

upper<-c[1,1]+1.96*c[1,4]/2
lower<-c[1,1]-1.96*c[1,4]/2

cindex<-rbind(cindex.orig,lower,upper)
cindex
dev.off()