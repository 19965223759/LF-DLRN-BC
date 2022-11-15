library(rmda)
library(rms)
library(caret)
raw_data<-read.csv("./nomogram_cox_clinical.csv",stringsAsFactors = FALSE,fileEncoding='GBK')

str(raw_data)

dd <- datadist(raw_data)
options(datadist="dd")
str(raw_data)

cox_multi<- as.formula(paste0('metastasis~', paste0(c('ki67','Histologic.type','PMI.T4','late_fusion_probability'),collapse='+')))
res_cph <- lrm(cox_multi,data=raw_data)
surv3<-function(x)1/(1+exp(-x))


res_nomo <- nomogram(res_cph ,lp=F,fun=surv3, # 10年
                     fun.at=c(.1,.2,.3,.4,.5,.6,.7,.8,.9),
                     funlabel='Risk of BC Metastasis')

plot(res_nomo)

#校正曲线


f5<- lrm(cox_multi, data=raw_data,x=TRUE,y=TRUE)
cal5<-calibrate(f5,method="boot",B=1000)


#单独画
jpeg("calibration-3.jpeg",width=150,height = 150,  units="mm",res=1000)
plot(1,type ="l",
     xlim =c(0,1),
     ylim =c(0,1),
     xlab = "Predicted Probability",
     ylab = "Observed Probability",
     legend = FALSE,
     subtitles = FALSE)

abline(0,1,col =topo.colors(4),lty = 1,lwd = 2)
lines(cal5[,c("predy","calibrated.corrected")], lty = 2,lwd = 2,col=heat.colors(5),pch =16)
dev.off()
