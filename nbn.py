dprint("Start Creating BernoulliNB Bayesian Network")
bnb = BernoulliNB()
y_pred = bnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0] ,(y_test != y_pred).sum()))

dprint("Get the parameters")
params = bnb.get_params(deep=True)
print params

dprint("Get the log probabilitites")
log_proba = bnb.score(X_test,y_test)
print log_proba