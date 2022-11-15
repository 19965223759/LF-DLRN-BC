library(rmda)

library(rms)
library(caret)
raw_data<-read.csv("./nomogram_cox_clinical.csv",stringsAsFactors = FALSE)
range(raw_data$DMFS)
str(raw_data)
model1<- decision_curve(metastasis~ki67+Histologic.type,data= raw_data,
                        family = binomial(link ='logit'),
                        thresholds= seq(0,1, by = 0.01),
                        confidence.intervals = 0.95,
                        study.design = 'case-control',
                        population.prevalence = 0.3)
model2<-decision_curve(metastasis~ki67+Histologic.type+PMI.T4,
                       data = raw_data,family = binomial(link ='logit'),
                       thresholds = seq(0,1, by = 0.01),
                       confidence.intervals= 0.95,
                       study.design = 'case-control',
                       population.prevalence= 0.3)
model3<-decision_curve(metastasis~ki67+Histologic.type+PMI.T4+late_fusion_probability, 
                       data = raw_data,family = binomial(link ='logit'),
                       thresholds = seq(0,1, by = 0.01),
                       confidence.intervals= 0.95,
                       study.design = 'case-control',
                       population.prevalence= 0.3)
List<- list(model1,model2,model3)
jpeg("DCA.jpeg",width=150,height = 150,  units="mm",res=1000)
plot_decision_curve(List,
                    curve.names=c('clinical','clinical+PMI','clinical+PMI+late_fusion_probability'),
                    cost.benefit.axis =FALSE,col= c('darkturquoise','gold','tomato'),
                    confidence.intervals=FALSE,
                    standardize = TRUE)

dev.off()