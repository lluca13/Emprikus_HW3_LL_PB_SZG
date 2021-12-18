# Demography Estimation Homework

## Objective
Reproduce the charts of the linked article and create an estimation of gender based on network features!

source data: https://snap.stanford.edu/data/soc-pokec.html \
original article: https://ericdongyx.github.io/papers/KDD14-Dong-et-al-WhoAmI-demographic-prediction.pdf

## Scoring

20 points is the max points, but 25 is achievable! The scoring is designed so that even people with little python 
experience can achiave at least 10

* 10 points:\
  Create an estimation of gender on the test set! Count the number of male and female friends of each node,
and predict accordingly
* 15 points:\
  recreate the chart 3 - 6 we discussed in class from the article. Make sure you write some analysis for
each chart. There is no embeddedness function in NetworkX, so you can skip that
* 20 points:\
  Create an estimation of the missing gender using the triangles in the network. You can use 
this resource: https://notebook.community/SubhankarGhosh/NetworkX/4.%20Cliques,%20Triangles%20and%20Graph%20Structures%20(Instructor)
  for counting triangles \
  This may be computationally intensive, and you can work around this any way you can! Reduce the size of the test set, 
  reduce the size of the whole graph, etc. A progress tracker such as `tqdm` may be really helpful here!\
  As a last step, evaluate the prediction against the known gender, and compare it with the prediction that only
  uses the neighbors. 
    
* +5 extra points: \
You can get these regardless of the rest of your points
  * do all of the above in .py files as you see in the repository instead of purely in a notebook
  * use clean code! Formatting, docstrings, sensible variable and function names
  * Structure your code well with functions
  * upload your solution to Gihub, and the only thing you have to send me is a link to a gh repo that contains the code
    

### Additional instructions

Make your life easier, and do the stuff for the extra points :)\
Make sure your charts are nice! Colors, title, scales, axislabels all matter\
Be sceptical about what you get! Make sure you do NOT include information that you shoulld not know in your estimations\
Please ask your questions on a forum that everyone can see, but if you need a quick answer, throw me an email at 
`karpatika92@gmail.com`

#### Good Luck!
    