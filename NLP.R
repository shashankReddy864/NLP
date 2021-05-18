library(quanteda)
library(ggplot2)
library(stopwords)
library(topicmodels)
library(tidytext)
library(dplyr)
library(caret)
library(pROC)
library(rpart)
library(rpart.plot)



df<- read.csv('gastext.csv',stringsAsFactors=F)

str(df)
summary(df)

# Establish the corpus and initial DFM matrix
myCorpus<-corpus(df$Comment)
summary(myCorpus)


myDfm <- dfm(myCorpus)
View(myDfm)

# Simple frequency analyses
tstat_freq <- textstat_frequency(myDfm)
head(tstat_freq,20)


# Visulize the most frequent terms
myDfm%>% textstat_frequency(n=20)%>%
  ggplot(aes(x=reorder(feature,frequency),y=frequency))+
  geom_point()+
  labs(x=NULL,y='Frequency')+
  theme_minimal()


# Remove stop words and perform stemming
myDfm<-dfm(myCorpus,remove_punc=T,
           remove=c(stopwords('english')),
           stem=T)

dim(myDfm)

# Removing terms like ( get, can) which doesn't make any significance 
mystopwords1<- c('get,use')
myDfm <- dfm(myCorpus,remove_punc = T,
             remove=c(stopwords('english'),mystopwords1),
             stem = T)

#Removing Productx as it very frequent
myDfm <- dfm(myCorpus,remove_punc = T,
             remove=c(stopwords('english'),mystopwords1, 'productx'),
             stem = T)


# Control sparse terms: to further remove some very infrequency words
myDfm<- dfm_trim(myDfm,min_termfreq=4, min_docfreq=2)
dim(myDfm)



textplot_wordcloud(myDfm,max_words=200)
topfeatures(myDfm,30)


# You can also explore other terms, such as "price" and "servic"
term_sim <- textstat_simil(myDfm,
                           selection="price",
                           margin="feature",
                           method="cosine")
as.list(term_sim,n=5)

term_sim1 <- textstat_simil(myDfm,
                            selection="servic",
                            margin="feature",
                            method="cosine")
as.list(term_sim1,n=5)



myDfm <- dfm_remove(myDfm, c('shower','point'))
myDfm <- as.matrix(myDfm)
myDfm <-myDfm[which(rowSums(myDfm)>0),]
myDfm <- as.dfm(myDfm)


# Topic Modeling
myLda <- LDA(myDfm,k=4,control=list(seed=101))
myLda


# Term-topic probabilities
myLda_td <- tidy(myLda)
myLda_td



# Visulize most common terms in each topic
top_terms <- myLda_td %>%
  group_by(topic) %>%
  top_n(8, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)


top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic)))+
  geom_bar(stat = "identity", show.legend =FALSE)+
  facet_wrap(~ topic, scales = "free")+
  coord_flip()

df[,3:15]<-lapply(df[,3:15],factor)
str(df)
dfSA <- df[,3:15]
str(dfSA)



# Create Training Data
set.seed(101)
trainIndex <- createDataPartition(dfSA$Target,
                                  p=0.7,
                                  list=FALSE,
                                  times=1)
dfSA.train <- dfSA[trainIndex,]


# Create Validation Data
dfSA.valid <-dfSA[-trainIndex,]


# Build a decision tree model
tree.model <- train(Target~.,
                    data=dfSA.train,
                    method="rpart",
                    na.action=na.pass)
# Display decision tree results
tree.model

# Display decision tree plot
prp(tree.model$finalModel,type=2,extra=106)

#Evaluation model performance using the validation dataset
prediction <- predict(tree.model,newdata=dfSA.valid,na.action = na.pass)

#Criteria 1: the confusion matrix
confusionMatrix(prediction,dfSA.valid$Target)

#Criteria 2: the ROC curve and area under the curve
tree.probabilities <- predict(tree.model,newdata=dfSA.valid,type='prob',na.action=na.pass)

tree.ROC <- roc(predictor=tree.probabilities$`1`,
                response=dfsm.valid$Target,
                levels=levels(dfSA.valid$Target))
plot(tree.ROC)
tree.ROC$auc

#We will first generate SVD columns based on the entire corpus
# Pre-process the training corpus
modelDfm <- dfm(myCorpus,
                remove_punc = T,
                remove=c(stopwords('english'),mystopwords1),
                stem = T)
dim(modelDfm)

# Further remove very infrequent words 
modelDfm<- dfm_trim(modelDfm,min_termfreq=4, min_docfreq=2)
dim(modelDfm)

# Weight the predictiv DFM by tf-idf
modelDfm_tfidf <- dfm_tfidf(modelDfm)
dim(modelDfm_tfidf)
library(quanteda.textmodels)
modelSvd <- textmodel_lsa(modelDfm_tfidf, nd=8)
head(modelSvd$docs)
dim(modelSvd$docs)


#Create a New Dataframe which stores both text and nontext information
dfSA_two<- cbind(dfsm,as.data.frame(modelSvd$docs))
summary(dfSA_two)
str(dfSA_two)


trainIndex2 <- createDataPartition(dfSA_two$Target,
                                   p=0.7,
                                   list=FALSE,
                                   times=1)
dfSA_two.train <- dfSA_two[trainIndex,]
dfSA_two.valid <-dfSA_two[-trainIndex,]


tree2.model <- train(Target~.,
                     data=dfSA_two.train,
                     method="rpart",
                     na.action=na.pass)
tree2.model
prp(tree2.model$finalModel,type=2,extra=106)

prediction2 <- predict(tree2.model,newdata=dfSA_two.valid,na.action = na.pass)

confusionMatrix(prediction2,dfSA_two.valid$Target)

tree2.probabilities <- predict(tree2.model,newdata=dfSA_two.valid,type='prob',na.action=na.pass)

tree2.ROC <- roc(predictor=tree2.probabilities$`1`,
                 response=dfSA_two.valid$Target,
                 levels=levels(dfSA_two.valid$Target))
plot(tree2.ROC)
tree2.ROC$auc

