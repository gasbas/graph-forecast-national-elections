from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import string
import itertools
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
 
models = {
    "A":['A', 'B', 'A B D'],
    "C":['C', 'C**H', 'C H', 'B+F', 'H p1 p2', 'C p1 p2', 'C/H',
          'F/H', 'D H p2', 'C D p2', 'C p2 p3', 'H p2', 'D F H',
          'H p2 p3', 'C F', 'C E F', 'H p1', 'C p1', 'F p3^2',
          'H p4', 'B C p4', 'A^3', 'D H p1', 'E H p1', 'C D p1',
          'F H', 'A G H', 'F p4^2', 'H p3'],
    "D":['D', 'B**p1', 'A/D', 'D p2 p3', 'D^2 F', 'C**D', 'C-F', 
           'C p4', 'A-F', 'C D p1', 'D p2', 'B**p3', 'D^2 p2', 'B D p1',
           'E p2^2', 'D p3 p5', 'H+p1', 'D p3', 'A/H', 'F^2 H', 'D p5',
           'C^2 p4', 'C D^2', 'B^2 D', 'D^2 p3', 'E p3 p5', 'E F^2',
           'A p2 p3', 'F p1 p3', 'C p1^2', 'A p1 p5', 'D^2 p5', 'D p1 p4', 'G+p4'],
    "E":['E', 'D-E', 'E^2', 'B D p1', 'B E p1', 'B D^2', 'D**E',
       'B D p2', 'E p2', 'B D p3', 'D^2 F', 'D E p3', 'D^2 p4',
       'E^2 F', 'B**p2', 'E p3 p4', 'D F', 'D p3', 'A-B', 'B G p4',
       'B E F', 'D E F', 'B E p2', 'B D p4', 'A/D'],
    "F":['F', 'E+F', 'D**F', 'D**E', 'E^2', 'F p1 p2', 'D p1 p2', 'D^3', 'D F p1',
       'E^2 p1', 'F p3 p4', 'D p3 p4', 'A+D', 'F^2 p1', 'E F p1', 'E^2 F', 'E^3',
       'F p1', 'D F p2', 'p1^3', 'A/D', 'E p1 p2', 'E p3 p4', 'A E H'],
    "G":['G', 'F+G', 'D-p1', 'E/G', 'D-H', 'A^2 B',
         'D**G', 'D**F', 'F p2 p3', 'G p2 p3', 'F**G', 'D^2',
         'F^2','p2+p3', 'G p1 p3', 'F p1 p3', 'G^2 p3'],
    "H":['F+H', 'D H', 'F^2 H', 'F-H', 'D**H', 'E**F', 'H p1 p3', 'B F p4', 'F/H', 'H p1',
           'B H p2', 'A**H', 'D**F', 'D H p1', 'H p2 p5', 'H p3 p4', 'F H p1', 'B H p1', 'D-G',
           'B H p3', 'F^2 p1', 'A B F', 'D H p3', 'C**H', 'H p3', 'E p4^2', 'B/H', 'B H p5', 'H p2',
           'D H p2', 'B G p2', 'E^2 p4', 'E F^2', 'H/p4', 'B/p4', 'D F H', 'D p3 p4', 'E p1 p2', 'F p1^2',
           'A H p1', 'F H p2', 'F H p3', 'F p3 p4', 'A D H', 'E F p2', 'A D p3', 'C p3 p4', 'B F p1',
           'E/p5', 'C p4^2', 'C D F', 'A H^2', 'A p3 p4', 'E p3 p4', 'B F^2', 'A D p1', 'G p3^2', 
           'G H p5', 'G H^2', 'D E p2', 'G p2', 'F','A G p4','H p1 p2','A D G',
          'D-E', 'H/p1', 'H', 'A C D', 'H p3^2', 'A D p2','A G p5', 'B E p4',
           'H p5', 'A H p4', 'B/F', 'E/F', 'C^2 p3','E H p5', 'C p1 p4', 'H/p3', 
           'A E p3', 'F p1 p4', 'E^2 G', 'C p1 p2', 'C E F', 'D E H', 'E p2 p3',
           'D p1 p5', 'D p4', 'H/p5', 'B F H', 'D^2 H', 'E p4 p5', 'E H^2', 'A B E',
           'B C p2', 'A E p2', 'B H','B^2 H', 'C p2 p3', 'E G^2', 'C H p3', 'B**F', 
           'C p2', 'H/p2'],
    "M1":['p1', 'F p1', 'A B p1', 'F^2 p1', 'p1^3', 'B/D', 'D F p1', 'B D p1', 'A D p1',
           'D^2 p1', 'D p1^2', 'F/p4', 'H p2 p5', 'D/p1', 'H/p2', 'p1/p4', 'E/F', 'G/H',
           'A/p5', 'F p2 p4', 'D^2 H', 'F H p1', 'F/G', 'A/D'],
    "M2":['p1+p2', 'F+p3', 'A+p3', 'B**p1', 'D+p4', 'B F p1', 'A/D', 'E p2 p3',
           'p5^2', 'D E F', 'F p2', 'A**p1', 'p1', 'D F^2', 'B D p2', 'F p4', 'B p1 p2',
           'F p3', 'D p3', 'D^2', 'A p1^2', 'D F p2', 'B F^2', 'p2^2 p3', 'A**p3', 'B p1 p4',
           'H p1 p4', 'F G p5', 'B G p3', 'F p5', 'D p4', 'H p4 p5', 'A p2^2', 'A C p1', 'D**F',
           'C/p4', 'p5^3', 'F^2 p2', 'G p1 p5', 'C/F', 'F p2 p4', 'D**H', 'B E p4', 'p2^2 p4',
           'A D p2', 'p1 p3 p4'],
    "P":['p1', 'F+p2', 'D p3', 'p1^2', 'D p1^2', 'A+p2', 'B F p1', 'p1**p2',
         'A p1', 'F p2^2', 'p4-p5', 'B p1 p4', 'D p1 p2', 'D^2 p1', 'p1 p2 p3', 'D^2 p3',
         'F p1^2', 'D F p1', 'p1 p2', 'D p1', 'A D p1', 'A B p2', 'p1**p3',
         'D p2', 'D F', 'F p2', 'p2 p3 p4', 'B D p4', 'F p3', 'H p1 p2'],
    "Q":['A', 'B', 'C', 'D','E', 'F', 'G', 'H', 'p1','p2', 'p3', 'p4', 'p5'],
    "R1":['D/p1', 'p1', 'F/p1', 'E p1^2', 'C^2 H'],
    "R2":['C F', 'p1', 'F H', 'p1 p2', 'B p1', 'D H', 'A D', 'B D', 'F G',
       'p1^2', 'E F', 'D E', 'A B', 'D^2', 'B F', 'D', 'C D', 'D F',
       'D G', 'A F', 'F^2', 'G H', 'D p2', 'A H', 'C H', 'p1 p5', 'F',
       'B p2', 'H p3', 'p2^2', 'p2', 'A G', 'A E', 'A C'],
    "R3":['p1+p2', 'D-p3', 'F-p4', 'F+p5', 'H+p4', 'C D p1', 'A-D', 'D p2 p5',
           'F-H', 'D E p1', 'p1^2 p2', 'D**p3', 'D p1 p2', 'F G p3', 'p2 p3^2',
           'D p2', 'D p1', 'D^3', 'F p1 p2', 'p1 p3^2', 'E**F', 'A^2 p1', 'D p4',
           'D p2^2', 'F-p3', 'D p3', 'D^2 F', 'A p2 p3', 'p2 p3', 'C F p3', 'p3 p4^2', 'D p1 p4'],
    "S":['A', 'B', 'C', 'D','E', 'F', 'G', 'H', 'p1','p2', 'p3', 'p4', 'p5'],
    "U":['D+F', 'D p1', 'D^2 p1', 'E H p2', 'B F p1', 'D p1^2', 'D**p1', 'F p4 p5',
         'A/D', 'D p2', 'E G p1', 'F p3', 'F p5', 'D F p5', 'D^3', 'H p1', 'D p2 p4',
         'D p1 p3', 'D p1 p2', 'B D H', 'H^2 p1', 'F^2 p3', 'A C p2', 'D p1 p5', 'p1 p2 p5',
         'A/p5', 'E/p4', 'D F p3', 'E F p1', 'E p4^2', 'C E G', 'C G p2', 'F^2 p5', 'E-H', 'A D p1',
         'A/p2', 'D H p3', 'F p1 p2', 'F p1 p4', 'H p1 p3', 'D p2 p3', 'B/p3', 'D H^2', 'H p1^2', 'p1^2',
         'B/p4', 'p1', 'F p1', 'B^2 p1', 'p1^3', 'E F H', 'A p1 p2', 'D F p1', 'F^2 p1', 'F H p5', 'A/B',
         'p2/p4', 'E p1 p3', 'E p2 p5', 'F/p1', 'B H p5', 'B H^2', 'E p1 p2', 'C/p5', 'D^2 p3', 'B H p4',
         'F H p4', 'B E p3', 'A p1 p3', 'D H p1', 'p2/p5', 'H/p5', 'C F H', 'C H p4', 'G p1 p4', 'F p5^2',
         'H^3', 'B-C', 'H p1 p2'],
    "W1":["p1","D/G","D F p1","A/p3"],
    "W2":["p1^2 p2","p1**p2","D-G","C+F"], 
    "W3":['A', 'B', 'C', 'D','E', 'F', 'G', 'H', 'p1','p2', 'p3', 'p4', 'p5'],
    'other':['p1^2', 'H', 'C', 'p2', 'A', 'D p1',
            'p1', 'F G', 'A p1', 'p3', 'E', 'p4',
            'p5', 'B', 'p1 p2', 'H p1', 'D', 'G^2',
            'F p1', 'F', 'D p2', 'D F p1', 'D H p1',
            'F p1^2', 'G H p1', 'D p1^2', 'A G p1', 'E p1',
            'F H p1', 'B D G', 'D^2 p1', 'C**H', 'A D p1']}
 
class Regressor(BaseEstimator):
    def __init__(self):
        
        self.modelA = MultiOutputRegressor(GradientBoostingRegressor())
        self.modelC = LinearRegression()
        self.modelD = LinearRegression()
        self.modelE = LinearRegression()
        self.modelF = LinearRegression()
        self.modelG = LinearRegression()
        self.modelH = LinearRegression()
        self.modelM1 = LinearRegression()
        self.modelM2 = LinearRegression()
        self.modelP = LinearRegression()
        self.modelQ = MultiOutputRegressor(GradientBoostingRegressor())
        self.modelR1 = RandomForestRegressor()
        self.modelR2 = MultiOutputRegressor(GradientBoostingRegressor())
        self.modelR3 = LinearRegression()
        self.modelS = LinearRegression()
        self.modelU = MultiOutputRegressor(GradientBoostingRegressor())
        self.modelW1 = DecisionTreeRegressor()
        self.modelW2 = DecisionTreeRegressor()
        self.modelW3 = MultiOutputRegressor(GradientBoostingRegressor())
        self.modelothers = LinearRegression()
        self.preprocessing = PolynomialFeatures(degree=3)
        self.alphabet = list(string.ascii_uppercase)
        
        targets=[]
        for i in range(len(self.alphabet)):
            targets+=[self.alphabet[i]+str(j) for j in range(1,81)]
        self.targets=targets

        self.cols_A = pd.Series(self.targets)[[i.startswith("A") for i in self.targets]].tolist()
        self.cols_C = pd.Series(self.targets)[[i.startswith("C") for i in self.targets]].tolist()
        self.cols_D = pd.Series(self.targets)[[i.startswith("D") for i in self.targets]].tolist()
        self.cols_E = pd.Series(self.targets)[[i.startswith("E") for i in self.targets]].tolist()
        self.cols_F = pd.Series(self.targets)[[i.startswith("F") for i in self.targets]].tolist()
        self.cols_G = pd.Series(self.targets)[[i.startswith("G") for i in self.targets]].tolist()
        self.cols_H = pd.Series(self.targets)[[i.startswith("H") for i in self.targets]].tolist()
        self.cols_M1 = pd.Series(self.targets)[[i.startswith("M") for i in self.targets]].tolist()[:16]
        self.cols_M2 = pd.Series(self.targets)[[i.startswith("M") for i in self.targets]].tolist()[16:]
        self.cols_P = pd.Series(self.targets)[[i.startswith("P") for i in self.targets]].tolist()
        self.cols_Q = pd.Series(self.targets)[[i.startswith("Q") for i in self.targets]].tolist()
        self.cols_R1 = pd.Series(self.targets)[[i.startswith("R") for i in self.targets]].tolist()[:16]
        self.cols_R2 = pd.Series(self.targets)[[i.startswith("R") for i in self.targets]].tolist()[16:32]
        self.cols_R3 = pd.Series(self.targets)[[i.startswith("R") for i in self.targets]].tolist()[32:]
        self.cols_S = pd.Series(self.targets)[[i.startswith("S") for i in self.targets]].tolist()
        self.cols_U = pd.Series(self.targets)[[i.startswith("U") for i in self.targets]].tolist()
        self.cols_W1 = pd.Series(self.targets)[[i.startswith("W") for i in self.targets]].tolist()[:16]
        self.cols_W2 = pd.Series(self.targets)[[i.startswith("W") for i in self.targets]].tolist()[16:31]
        self.cols_W3 = pd.Series(self.targets)[[i.startswith("W") for i in self.targets]].tolist()[31:]
        self.cols_others =  list(set(self.targets)-set(self.cols_A)-set(self.cols_C)-set(self.cols_D)-set(self.cols_E)-set(self.cols_F)-set(self.cols_G)-set(self.cols_H)-set(self.cols_M1)-set(self.cols_M2)-set(self.cols_P)-set(self.cols_Q)-set(self.cols_R1)-set(self.cols_R2)-set(self.cols_R3)-set(self.cols_RS)-set(self.cols_RU)-set(self.cols_W1)-set(self.cols_W2)-set(self.cols_W3))
                                                                   
   	def features(self,X):
        ls_col=['A', 'B', 'C', 'D','E', 'F', 'G', 'H', 'p1','p2', 'p3', 'p4', 'p5']
        X_train2=self.preprocessing.fit_transform(X)
        X_train2=pd.DataFrame(X_train2,columns=self.preprocessing.get_feature_names(ls_col))
        combi=itertools.combinations(ls_col, 2)
        X=pd.DataFrame(X,columns=ls_col)
        for v1,v2 in combi:
            X["{}+{}".format(v1,v2)]=X[v1]+X[v2]
            X["{}-{}".format(v1,v2)]=X[v1]-X[v2]
            X["{}/{}".format(v1,v2)]=X[v1]/X[v2]
            X["{}**{}".format(v1,v2)]=X[v1]**X[v2]
            
        X = pd.concat([X,X_train2],axis=1)
        return X
        
    def fit(self, X, Y):
        X = self.features(X)
        
        Y = pd.DataFrame(Y,columns=self.targets)

        self.modelA.fit(X, Y[self.cols_A])
        self.modelC.fit(X[models["C"]], Y[self.cols_C])
        self.modelD.fit(X[models["D"]], Y[self.cols_D])
        self.modelE.fit(X[models["E"]], Y[self.cols_E])
        self.modelF.fit(X[models["F"]], Y[self.cols_F])
        self.modelH.fit(X[models["H"]], Y[self.cols_H])
        self.modelM1.fit(X[models["M1"]], Y[self.cols_M1])
        self.modelM2.fit(X[models["M2"]], Y[self.cols_M2])
        self.modelP.fit(X[models["P"]], Y[self.cols_P])
        self.modelQ.fit(X[models["Q"]], Y[self.cols_Q])
        self.modelR1.fit(X[models["R1"]], Y[self.cols_R1])
        self.modelR2.fit(X[models["R2"]], Y[self.cols_R2])
        self.modelR3.fit(X[models["R3"]], Y[self.cols_R3])
        self.modelS.fit(X[models["S"]], Y[self.cols_S])
        self.modelU.fit(X[models["U"]], Y[self.cols_U])
        self.modelW1.fit(X[models["W1"]], Y[self.cols_W1])
        self.modelW2.fit(X[models["W2"]], Y[self.cols_W2])
        self.modelW3.fit(X[models["W3"]], Y[self.cols_W3])
        self.modelothers.fit(X[models["other"]], Y[self.cols_others])
        
    def predict(self, X):
        
        #X = X / X.max()      
        X = self.features(X)
        pred = pd.DataFrame(columns=self.targets,index=list(range(X.shape[0])))
        pred.loc[:,self.cols_A]=self.modelA.predict(X[models["A"]])
        pred.loc[:,self.cols_C]=self.modelC.predict(X[models["C"]])
        pred.loc[:,self.cols_D]=self.modelD.predict(X[models["D"]])
        pred.loc[:,self.cols_E]=self.modelE.predict(X[models["E"]])
        pred.loc[:,self.cols_F]=self.modelF.predict(X[models["F"]])
        pred.loc[:,self.cols_G]=self.modelG.predict(X[models["G"]])
        pred.loc[:,self.cols_H]=self.modelH.predict(X[models["H"]])
        pred.loc[:,self.cols_M1]=self.modelM1.predict(X[models["M1"]])
        pred.loc[:,self.cols_M2]=self.modelM2.predict(X[models["M2"]])
        pred.loc[:,self.cols_P]=self.modelP.predict(X[models["P"]])
        pred.loc[:,self.cols_Q]=self.modelQ.predict(X[models["Q"]])
        pred.loc[:,self.cols_R1]=self.modelR1.predict(X[models["R1"]])
        pred.loc[:,self.cols_R2]=self.modelR2.predict(X[models["R2"]])
        pred.loc[:,self.cols_R3]=self.modelR3.predict(X[models["R3"]])
        pred.loc[:,self.cols_S]=self.modelS.predict(X[models["S"]])
        pred.loc[:,self.cols_U]=self.modelU.predict(X[models["U"]])
        pred.loc[:,self.cols_W1]=self.modelW1.predict(X[models["W1"]])
        pred.loc[:,self.cols_W2]=self.modelW2.predict(X[models["W2"]])
        pred.loc[:,self.cols_W3]=self.modelW3.predict(X[models["W3"]])
        pred.loc[:,self.cols_others]=self.modelothers.predict(X[models["other"]])
        
        return pred.values