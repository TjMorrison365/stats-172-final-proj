rm(list = ls())
library(ggplot2)
library(pROC)
library(randomForest)
library(rpart)
library(tidytext)
library(topicmodels)
library(dplyr)
library(reshape2)

# Y variable "Gross"
movies = read.csv("IMDB Movies Dataset_exported.csv", stringsAsFactors=FALSE)
str(movies)
summary(movies)
View(movies)

##### Clean Data #####
# 1. We want to remove "Series_Title" and Poster_Link" as it is an indentifier variable.
# THIS WILL BE DONE CUMULATIVELY THE END


# 2. We want to rename certificates so that the categories have significant
# observation sizes and interpretable names.
summary(movies$Certificate)

# GP is PG renamed so all GP will change to PG
movies$Certificate[movies$Certificate == "GP"] = "PG"

# U/A is UA so all U/A will change to UA
movies$Certificate[movies$Certificate == "U/A"] = "UA"

# Change in times made the rating 16 turn to R so we will lump them all as R
movies$Certificate[movies$Certificate == "16"] = "R"

# Approved is Passed renamed so we will change Approved to Passed
movies$Certificate[movies$Certificate == "Approved"] = "Passed"

# Everything without a label was "Denyed" and therefore -> unrated.
movies$Certificate[movies$Certificate == ""] = "Unrated"

# I will place TV categories in the unrated bin because all the different TV
# categories only account for 5 observations and "Unrated" movies/tv are generally
# treated as "unknown" and because of that have flexiblility in their interpretations.
movies$Certificate[movies$Certificate %in% c("TV-PG", "TV-14", "TV-MA")] = "Unrated"

# G, PG, PG-13, R, and U, UA, A, and Passed, Unrated.
summary(movies$Certificate)


# 3. We want to remove " min" from runtime to turn it into a numeric variable.
movies$Runtime = gsub(" min", "", movies$Runtime)
movies$Runtime = as.numeric(movies$Runtime)
str(movies)

# We may want to sepereate "Genre" variable so that each genre can be read as a factor.
# (Most likely by using dummy variables, ex. Action = {0,1})
# Genre List: Drama, Crime, Action, Adventure, Sci-Fi, Western, Biography, Comedy, Thriller, Animation
# Family, Mystery, Music, War, History, Romance, Horror, Sport, Film-Noir, al
# Below is just the method used to get all genre's quickly
temp = gsub(",", "", movies$Genre)
temp = gsub("Drama", "", temp)
temp = gsub("Crime", "", temp)
temp = gsub("Action", "", temp)
temp = gsub("Adventure", "", temp)
temp = gsub("Sci-Fi", "", temp)
temp = gsub("Western", "", temp)
temp = gsub("Biography", "", temp)
temp = gsub("Comedy", "", temp)
temp = gsub("Thriller", "", temp)
temp = gsub("Animation", "", temp)
temp = gsub("Family", "", temp)
temp = gsub("Fantasy", "", temp)
temp = gsub("Mystery", "", temp)
temp = gsub("Music", "", temp)
temp = gsub("War", "", temp)
temp = gsub("History", "", temp)
temp = gsub("Romance", "", temp)
temp = gsub("Horror", "", temp)
temp = gsub("Sport", "", temp)
temp = gsub("Film-Noir", "", temp)
temp = gsub("al", "", temp)
temp = gsub("  ", "", temp)
temp = gsub("   ", "", temp)
temp = gsub("    ", "", temp)
temp = gsub("     ", "", temp)
temp = gsub("      ", "", temp)
unique(temp)

for (genre_name in c("Drama", "Crime", "Action", "Adventure", "Sci-Fi",
                     "Western", "Biography", "Comedy", "Thriller", "Animation",
                     "Family", "Mystery", "Music", "War", "History", "Romance",
                     "Horror", "Sport", "Film-Noir", "al")) {
  movies[[gsub("-", "", genre_name)]] = ifelse(grepl(genre_name, movies$Genre), 1, 0)
  print(paste0(genre_name, ": ", sum(movies[[genre_name]])))
}
View(movies)

# Now we can drop "Genre"
# THIS WILL BE DONE CUMULATIVELY THE END

# 4. Find unique ways to deconstruct "Overview" (length, keyword use)
##### Text Mining #####

#I'm creating a new, tokenized data frame that is called "tokens"
#(not very creative, but it works)
tokens = movies %>% unnest_tokens(word, Overview)

#load stop_words dataframe
data(stop_words)
#remove all rows consisting of a stop word
tokens_clean = tokens %>% anti_join(stop_words)

#first, count the number of times each word shows up
tokens_count = tokens_clean %>%
  #sorts from most frequent to least
  count(word, sort = TRUE) %>%
  #reorders the factor levels for the plot
  mutate(word = reorder(word,n))

#view the first 10 words:
ggplot(data = tokens_count[1:10,]) +
  geom_col(aes(x=word, y=n)) +
  labs(x = "Word", y = "Count")+
  coord_flip()

#Previously we counted the overall number of times each word showed up
#now we need to count the number of times each word shows up
#in each statement#note that Series_Title is an identifier for each comment
tokens_count <- tokens_clean %>%
#include id so it counts within unique id
count(Series_Title, word, sort = TRUE)%>%ungroup()

#tokens_count is a tidy data frame
#to do LDA, we need what is called a "Document Term Matrix" or DTM
dtm <- tokens_count %>%cast_dtm(Series_Title, word, n)

#now we can perform LDA! need to specify the number of topics (k)
#this is subjective and it's a good idea to try out a few
#different k values and see what "makes most sense"
#"makes most sence": are your topics meaningful? well-defined?
#start with k = 10
lda <- LDA(dtm, k = 8, control = list(seed = 1234))
#Let's first look at what our topics are!
#This essentially means looking at the most characteristic
#words for each topic. A plot is the best way to do this
#(in my opinion)

#beta matrix gives the per topic per word probabilities
#a higher beta means that word is important for that topic
topics <- tidy(lda, matrix = "beta")
#get a small data frame of the top 10 words for each topic
top_terms <- topics %>%group_by(topic) %>%
  #within each topic do some function
  top_n(10, beta) %>%
  #that function is take the top 10 in terms of beta
  ungroup() %>%arrange(topic, -beta)
#order (just for the plot)


top_terms %>%mutate(term = reorder(term, beta)) %>%
  #reorder term/word for plot
  ggplot() +
  geom_col(aes(x=term,y= beta, fill = factor(topic)),show.legend = FALSE) + #like geom_bar but more flexible
  facet_wrap(~ topic, scales = "free") +
  labs(c()) +
  coord_flip() #make words easier to read


#per-document-per-topic probabilities 
documents <- tidy(lda, matrix = "gamma")

documents_w<- documents %>%   
  select(document, topic, gamma) %>%
  dcast(document ~ topic, value.var = "gamma")

colnames(documents_w) <- c("Series_Title", paste0("Topic", c(1:8)))
movies_lda <- merge(documents_w, movies,  by="Series_Title", all = T)

str(movies_lda)

#if some comments didn't have text,  we won't have any probabilities
#for any of the topics - we'll remove them

movies_lda <- subset(movies_lda, !is.na(Topic1))
movies = movies_lda
View(movies_lda)

#PS: note a random forest could be used to figure out which of these topics should
#be included here!

##### Clean Data Cont' #####
# Then we'll want to drop Overview
# THIS WILL BE DONE CUMULATIVELY THE END

# 5. What does "Meta_score", and "No_of_Votes" mean?
# Meta_score - The meta score makes up reviews from the world's top critics (0-100)
# No_of_votes - How many ordianry IMDB users have rated the movie in total. AKA
# audience critic engagement.

# 6. Change Gross to numeric, Need to remove empty Y variable rows from
# test/training datasets. Make "Gross" a numeric variable.
movies$Gross = gsub(",", "", movies$Gross)
movies = movies[!movies$Gross=="",]
movies$Gross = as.numeric(movies$Gross)

# Find reasonable cutoff for a 'successful' movie cutoff in order to set a
# positive event.

summary(movies$Gross)

# Because the goal is to find "successful gross" and successful gross is neither
# properly defined and most likely to have a Gamma distrubution (where there
# exists no negatives and you can have extreme right skewness), I think it would
# be appropriate to say successful gross is a gross larger than average.
# Although the average would be effected by high positive outliers, I think they
# still contribute a good portion to predictive modeling as it's not entirely
# unlikely to find new movies fall into these higher grossing movie categories.

movies$High_Gross = ifelse(movies$Gross > mean(movies$Gross), 1, 0)
print(paste0(mean(movies$High_Gross)*100, "% of the data is a Successful Event (AKA is a Movie who's gross was greater than average)"))

# 7. Reduce the variability in Director (only take the most frequent directors
# and lump the rest in Unknown)
summary(as.factor(movies$Director))
movies$Director[!(movies$Director %in% c("Steven Spielberg", "Martin Scorsese", "Alfred Hitchcock", "Christopher Nolan",
    "Clint Eastwood", "David Fincher", "Quentin Tarantino", "Woody Allen",
    "Hayao Miyazaki", "Rob Reiner", "Billy Wilder", "Joel Coen"))] = "Unknown"

# 8. Create actor dummy variables for most prominent actors.
temp = append(movies$Star1, movies$Star2)
temp = append(temp, movies$Star3)
temp = append(temp, movies$Star4)
summary(as.factor(temp))


for (actor_name in c("Robert De Niro", "Tom Hanks", "Al Pacino", "Brad Pitt",
                     "Christian Bale", "Clint Eastwood", "Leonardo DiCaprio",
                     "Matt Damon", "Denzel Washington", "Ethan Hawke", "Johnny Depp",
                     "Scarlett Johansson")) {
  movies[[gsub(" ", "", actor_name)]] = ifelse(movies$Star1 == actor_name |
                                          movies$Star2 == actor_name |
                                          movies$Star3 == actor_name |
                                          movies$Star4 == actor_name, 1, 0)
}

# Drop the actor columns and gross
# THIS WILL BE DONE CUMULATIVELY THE END

# 9. Set Release Year NA to mode, and Meta Score NA to median
any(is.na(movies$Released_Year))
any(is.na(movies$Meta_score))


movies$Released_Year = as.factor(movies$Released_Year)
movies$Released_Year = as.character(movies$Released_Year)
movies$Released_Year[is.na(movies$Released_Year)] = 2014

summary(movies$Meta_score)
movies$Meta_score[is.na(movies$Meta_score)] = 78


# Final: Remove unnecessary columns and set categoricals as factors
clean_full_movies = movies
movies[ ,c("Series_Title", "Poster_Link", "Genre", "Overview", "Star1", "Star2", "Star3", "Star4", "Gross")] = list(NULL)
movies$Released_Year = as.numeric(movies$Released_Year)
for (column_name in colnames(movies)[!(colnames(movies) %in% c("Topic1", "Topic2", "Topic3", "Topic4", "Topic5", "Topic6", "Topic7", "Topic8", "Released_Year", "Runtime", "IMDB_Rating", "Meta_score", "No_of_Votes"))]) {
  movies[[column_name]] = as.factor(movies[[column_name]])
}
str(movies)

##### Forrest Model #####

### --------- Data Preparation

RNGkind(sample.kind = "default")
set.seed(2291352)
train.idx = sample(x = 1:nrow(movies), size = .8*nrow(movies))
train.df = movies[train.idx,]
test.df = movies[-train.idx,]

### --------- Tuning your forest

str(train.df)
set.seed(2291352)
mtry <- c(1:ncol(movies)-1) #What is reasonable (usually around sqrt(k)) (ncol()-1 is all x variables)
n_reps <- 10 # how many times do you want to fit each forest? for averaging
#make room for m, OOB error
keeps <- data.frame(m = rep(NA,length(mtry)*n_reps),#NOTE DIFFERENCE
                     OOB_err_rate = rep(NA, length(mtry)*n_reps))#NOTE DIFFERENCE
j = 0 #initialize row to fill#NOTE DIFFERENCE
for (rep in 1:n_reps){#NOTE DIFFERENCE
  print(paste0("Repetition = ", rep))#NOTE DIFFERENCE
  for (idx in 1:length(mtry)){
    j = j + 1 #increment row to fill over double loop#NOTE DIFFERENCE
    tempforest<- randomForest(High_Gross ~ .,
                              data = train.df, 
                              ntree = 1000, #fix B at 1000!
                              mtry = mtry[idx]) #mtry is varying
    #record iteration's m value in j'th row
    keeps[j , "m"] <- mtry[idx]#NOTE DIFFERENCE
    #record oob error in j'th row
    keeps[j ,"OOB_err_rate"] <- mean(predict(tempforest)!= train.df$High_Gross)#NOTE DIFFERENCE
    
  }
}

#calculate mean for each m value
keeps2 = keeps %>% 
  group_by(m) %>% 
  summarise(mean_oob = mean(OOB_err_rate))


#plot you can use to justify your chosen tuning parameters
ggplot(data = keeps2) +
  geom_line(aes(x=m, y=mean_oob)) + 
  theme_bw() + labs(x = "m (mtry) value", y = "OOB error rate") +
  scale_x_continuous(breaks = c(1:ncol(movies)-1))

tuned_m = keeps2$m[keeps2$mean_oob == min(keeps2$mean_oob)]
tuned_m = min(tuned_m)

# My results suggest an m of _ would be ideal for minimizing  OOB error.
#Note for Follet~ since the results of this could change and I have the random
# element in my final forest, my numbers may be different from what you got
set.seed(2291352)
final_forest = randomForest(High_Gross ~ .,
                            data = train.df,
                            ntree = 1000,
                            mtry = tuned_m, # based on tuning exercise
                            importance = TRUE)

final_forest

### ---------- RESULTS

#In most problems our goal emphasizes (1) prediction or (2)
#interpretation/description usually, people want value from both 1+2

### ---------- interested in prediction
#Start with ROC curve
pi_hat = predict(final_forest, test.df, type = "prob")[,"1"] #We extract our positive event
rocCurve = roc(response = test.df$High_Gross,
               predictor = pi_hat,
               levels = c("0", "1"))

plot(rocCurve, print.thres = TRUE, print.auc=TRUE)

# If we set pi* = 0.302, we are guarenteed a specificity of 0.763
# and sensitivity of 0.837.

# That is, we predict an unsuccessful gross 76.3% of the time when the movie is
# actually unsuccessful, and we successful gross 83.7% of the time when the movie is
# actually successful.

# AUC is 0.857

# Make a column of predicted values (Successful or Unsuccessful)
pi_star = coords(rocCurve, "best", ret = "threshold")[1,1]
test.df$forest_pred = ifelse(pi_hat > pi_star, 1, 0)

##### Forrest Model Notes #####
#
# I will need to redo interpretations when I include overview variables
# Done ^

##### Creating a GLM #####
#Random Forests provide something called Variable Importance Plots
#Plot ranks variables from most important to least important in terms of
# OUT OF SAMPLE prediction preformance

varImpPlot(final_forest, type = 1, main=("Final Forest"), bg = "black")
# Note: We can only run this code if importance = TRUE in our final_forest

cor(cbind(train.df$High_Gross, train.df$No_of_Votes, train.df$Adventure, train.df$Released_Year, train.df$IMDB_Rating))

test.df$forest_pred = as.factor(test.df$forest_pred)

my_cols <- c("green", "darkgreen")  
pairs(test.df[,c("High_Gross", "No_of_Votes", "Released_Year", "IMDB_Rating")], pch = 19,
      col = my_cols[test.df$forest_pred],
      upper.panel=NULL)

# Random forest pro: we get automatic variable selection, importance ranking.
# Random forest con: no directional effects (like correlation), lack interpretability.
# logistic regression pro: get nice directional effects in terms of odds.
# logistic regression con: no automatic variable selection.

# let's fit a logistic regression with only the "best" variables from the
# random forest. First create a bernoulli rv

m1 = glm(movies$High_Gross ~ No_of_Votes,
         data = movies, family = binomial(link = "logit"))
AIC(m1) #753.9967

m2 = glm(movies$High_Gross ~ No_of_Votes + Adventure,
         data = movies, family = binomial(link = "logit"))
AIC(m2) #722.7067, AIC decreases. Worth the added variable

m3 = glm(movies$High_Gross ~ No_of_Votes + Adventure + Director,
         data = movies, family = binomial(link = "logit"))
AIC(m3) #698.5767, AIC decreases. Worth the added variable

m4 = glm(movies$High_Gross ~ No_of_Votes + Adventure + Director + Released_Year,
         data = movies, family = binomial(link = "logit"))
AIC(m4) #692.208

m5 = glm(movies$High_Gross ~ No_of_Votes + Adventure + Director + Released_Year +
           Certificate,
         data = movies, family = binomial(link = "logit"))
AIC(m5) #658.7878

m6 = glm(movies$High_Gross ~ No_of_Votes + Adventure + Director + Released_Year  +
           Certificate + Crime,
         data = movies, family = binomial(link = "logit"))
AIC(m6) #656.1248

m7 = glm(movies$High_Gross ~ No_of_Votes + Adventure + Director + Released_Year  +
           Certificate + Crime + IMDB_Rating,
         data = movies, family = binomial(link = "logit"))
AIC(m7) #594.9087

m8 = glm(movies$High_Gross ~ No_of_Votes + Adventure + Director + Released_Year  +
           Certificate + Crime + IMDB_Rating + MattDamon,
         data = movies, family = binomial(link = "logit"))
AIC(m8) #585.372

m9 = glm(movies$High_Gross ~ No_of_Votes + Adventure + Director + Released_Year  +
           Certificate + Crime + IMDB_Rating + MattDamon + TomHanks,
         data = movies, family = binomial(link = "logit"))
AIC(m9) #575.6915

m10 = glm(movies$High_Gross ~ No_of_Votes + Certificate + Director + Adventure  +
            Certificate + Crime + IMDB_Rating + MattDamon + TomHanks + Topic1,
         data = movies, family = binomial(link = "logit"))
AIC(m10) #574.7047

m11 = glm(movies$High_Gross ~ No_of_Votes + Certificate + Director + Adventure  +
            Certificate + Crime + IMDB_Rating + MattDamon + TomHanks + Topic1 +
           Topic3,
         data = movies, family = binomial(link = "logit"))
AIC(m11) #574.0654 Although the AIC is getting lower, its not reducing by much
# In this instance I will use m9 for simplicity.

summary(m9)
#no standard errors greater than 5! no signs of complete separation
#collect beta vector
beta = coef(m9)
beta
#exponentiate to get odds ratios
exp(beta)
#ex) Interpret opponent rank coefficient:

#Holding all other movie characteristics constant,
#the odds of a given movie having successful gross increases
#by a factor of exp(6.555744e-06*10000) = 1.067754 for
#every 10k more IMDB votes.
exp(6.555744e-06*10000)
#That is, its odds increase by about 6.7%.

confint(m9)


#Wald confidence intevals for beta would be
#beta hat +/- sqrt(Variance(beta hat))*quantile
#or, 
betahat <- coef(m9)#coefficient estimates
var_betahat <- vcov(m9) %>%diag #diagonal of covariance matrix
quantile <- qnorm(.975, 0,1)
lower <- betahat - sqrt(var_betahat)*quantile
upper <- betahat + sqrt(var_betahat)*quantile

#these would need to be exponentiated to get odds ratio CI's
cbind(exp(lower),exp(upper))

##### Creating a GLM Notes #####
#
# I will need to redo interpretations when I include overview variables
# I will probably make the AIC scoring a loop
#

##### Rubric #####

# Visualization
# Outstanding (30)
# Modern and quality statistical graphics using modern R packages are used to enhance
# analysis.Thoughtful color palettes, labels, titles, and plot types. This helps the
# audience better understand the data and the problem at hand. Visualization is ultimately
# tied to methods chosen.

# Predictive Methods (I need to write new interpretations)
# Outstanding (30)
# Random forest is correctly tuned, interpreted, and used in conjunction with descriptive
# (GLM)analysis. Tuning is described appropriately (all the transparency) to support final
# predictive model. Out-of-sample (testing) metrics are appropriate and meaningfully
# described in the context of the data and research question. Code automation is fully
# incorporated.

# Data Mining Enhancements Finished (I need to label topics and tune number of topics)
# Outstanding (15)
# Project makes use of additional data mining / reproducibility techniques in a way that
# is appropriate, relevant to the problem, and creative.

# Presentation
# Outstanding (10)
# Information is verbally presented clearly and professionally. Presentation materials
# (maybe a”Powerpoint”, maybe not!) are easy to read and professional.

# GLM Model Choice and Justification
# Outstanding (5)
# All components of the final/chosen GLM are correctly presented. Random component choice is
# justified from a practical perspective (e.g,. ”this is binary data so...”) and statistical
# perspective (e.g., ”a lowerBIC suggested...”). Likewise for the link function. Method for
# choosing x variables is based on modern methods(e.g., random forest, clustering, text
# mining) and intuition/common sense/problem at hand.

# GLM Insights
# Outstanding (5)
# Interpretations of coefficients corresponding to both numeric and categorical variables
# are both technically correct and meaningful. Interpretations are tied back to original
# goals of the client. Confidence intervals are correctly and meaningfully explained in
# context and used to quantify the uncertainty around the interpretations.

# Engagement
# Outstanding (5)
# Writes a critique on 3 other students’ videos. This critique should include (1) your
# favorite aspectof the analysis presented (2) in what ways the analysis was/wasn’t helpful
# for their client in terms of actionable insights.



