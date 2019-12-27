import numpy as np
import pandas as pd







def weights(a, vi_bar, votes, weight_denominator,j):
    w = 0

    temp = votes.loc[a,:]
    num = temp.dot(votes.T)
    den = np.sqrt(weight_denominator[a]*weight_denominator)
    K = 1/np.abs(np.sum(num.div(den)))
    w = (num.div(den).fillna(0)).dot(votes[j].T)
    
    SeT = w*K
    return SeT



def prediction(a,j, votes, vi_bar, weight_denominator):
#    a, j = test.iloc[i,:].UserID, test.iloc[i,:].MovieID
    va_bar = vi_bar[vi_bar["UserID"] == a].values[:,-1][0]
#    print(w_dict)
#    if a not in w_dict.keys():
##        print("yoo")
#        w_dict[a] = {}
    try:
      SeT = weights(a, vi_bar, votes, weight_denominator, j)
  #    else:
  #        w = 0
  #        for i in votes.index:
  #            w += w_dict[a][i]*votes.loc[i,j]
  #        SeT = w_dict[a]['K']*w
          
      p = va_bar + SeT 
      if np.isnan(p):
        p = va_bar
    except:
      p = va_bar
    return p


def RMSE(Predictions,test):
    rm = np.sqrt(mean_squared_error(test["Rating"], Predictions))
    return rm

def MAE(Predictions,test):
    ma = mean_absolute_error(test["Rating"], Predictions)
    return ma





if __name__ == "__main__":
    
    print("Preparing data")
    
    train = pd.read_csv('TrainingRatings.txt', sep=",", header=None, names=["MovieID", "UserID", "Rating"], engine = "python")
    test = pd.read_csv('TestingRatings.txt', sep=",", header=None, names=["MovieID", "UserID", "Rating"], engine = "python")

    vi_bar = train.groupby(["UserID"])["Rating"].mean().reset_index()
    v = train.merge(vi_bar, on = "UserID")
    v["Minus"] = v["Rating_x"] - v["Rating_y"]
    votes = v.pivot("UserID", "MovieID", "Minus").fillna(0)

    weight_denominator = np.sum(np.square(votes), axis = 1)

    print("All set with data")


    Predictions = np.zeros(len(test))
    
       
    for q in range(len(test)):
        a, j = test.iloc[q,:].UserID, test.iloc[q,:].MovieID
        Predictions[q] = prediction(a,j, votes, vi_bar, weight_denominator)
        print(q,Predictions[q])
        
    from sklearn.metrics import mean_squared_error
    rmse = RMSE(Predictions,test)
    print("The RMSE of the prediction: ", rmse)
    
    from sklearn.metrics import mean_absolute_error
    mae = MAE(Predictions,test)
    print("The Mean Absolute Error: ", mae)    
