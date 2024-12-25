import os
import numpy as np
import pandas as pd
import argparse
import copy
import scipy.special
from utils import myDataset, torchDataset, set_args
from src.preprocess import run as runReg

np.set_printoptions(suppress=True, precision=6)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class DecisionMakingData():
    def __init__(self, args, seed=2024):
        self.name    = args.name
        self.a_nonmn = args.a_nonmn
        self.cvalue  = args.cvalue
        self.vvalue  = args.vvalue
        self.t_param = args.t_param
        self.r_param = args.r_param
        self.a_param = args.a_param
        self.y_param = args.y_param
        self.c_param = args.c_param
        self.v_param = args.v_param
        self.rng  = np.random.default_rng(seed)
        self.set_config()
        
    def set_config(self, dim=3):
        self.cov = 0.3
        self.dim = dim
        self.column = ['t','y','u0','u1','c1','c2','c3','P'] + ['x' + str(i) for i in range(self.dim)] 
        self.set_params()
        self.get_covariance()
    
    def set_params(self):
        self.psi = self.rng.uniform(-1.0, 1.0, size=(self.dim, 1)).round(1)
        self.csi = self.rng.uniform(-1.0, 1.0, size=(self.dim, 1)).round(1)
        self.vsi = self.rng.uniform(-1.0, 1.0, size=(self.dim, 1)).round(1)
        self.phi = self.rng.uniform(-1.0, 1.0, size=(self.dim, 3)).round(1)

    def get_covariance(self):
        matrix = np.full((self.dim, self.dim), float(self.cov), dtype=float)
        np.fill_diagonal(matrix, 1)
        self.matrix = matrix
    
    def get_x(self, num):
        return self.rng.multivariate_normal(mean=np.zeros(self.dim), cov=self.matrix, size=num)

    def get_cost(self, X):
        zh, zs = self.c_param

        z = np.cos(X) @ self.csi + X[:,1:] * X[:,:-1] @ self.csi[:-1]
        C = (sigmoid(z))*zs+zh

        if self.cvalue <= 0.0:
            return C
        else:
            C = C - C + self.cvalue
            return C
        
    def get_income(self, X):
        zh, zs = self.v_param
        z = np.cos(X) @ self.vsi + X[:,1:] * X[:,:-1] @ self.vsi[:-1]
        C = (sigmoid(z))*zs+zh

        if self.vvalue <= 0.0:
            return C
        else:
            C = C - C + self.vvalue
            return C

    def testing(self, num=500, rep=40, ifprint=True):
        X_ = self.get_x(num)
        X = np.tile(X_, (rep, 1))
        V = self.get_income(X)
        C = self.get_cost(X)

        pi2t = self.get_pi2t(X)
        T = self.rng.binomial(1, pi2t)

        pi2r = self.get_pi2r(X)
        R = self.rng.binomial(1, pi2r)

        pi2y0 = self.get_pi2y0(X)
        R2 = self.rng.binomial(1, pi2y0)

        if self.a_nonmn > 0:
            pi2a = self.get_pi2a(X)
            if ifprint: self.shows(pi2a)
            A1 = self.rng.binomial(1, pi2a)
            Y0, Y1 = copy.deepcopy(R2), copy.deepcopy(R2)
            Y0[R==1] = 1-copy.deepcopy(A1[R==1])
            Y1[R==1] = copy.deepcopy(A1[R==1])
        else:
            Y1 = np.maximum(R, R2)
            Y0 = (R2 - R) == 1

        Y = Y1 * T + (1-T)*Y0

        if self.a_nonmn > 0:
            PY01 = pi2r*pi2a
            PY10 = pi2r*(1-pi2a)
        else:
            PY01 = pi2r
            PY10 = pi2r - pi2r
        PY00 = (1-pi2r)*(1-pi2y0)
        PY11 = (1-pi2r)*pi2y0

        print('P1: {}, P0: {}, PR=1: {}, tau: {}, P3: {}.'.format(np.mean((R)), np.mean((Y1-Y0)==0), np.mean((Y1-Y0)==1), np.mean(PY01), np.mean((Y1-Y0)==-1)))

        Py0  = PY10 + PY11
        Py1  = PY01 + PY11

        df = pd.DataFrame(np.concatenate([X, C, C, V, V, T, Y, Y0, Y1, pi2t, PY01, pi2y0, Py0, Py1, PY01, PY00,PY11], axis=1), columns=self.column)
        return df, myDataset(df)

    def get_pi2t(self, X):
        sh, sc = self.t_param
        if self.name == 'linear':
            z = X @ self.psi
        elif self.name == 'toy':
            z = X @ self.psi + (X[:,1:] + X[:,:-1])**2 @ self.psi[:-1] / self.dim
        elif self.name == 'sin':
            z = X @ self.psi + (X[:,1:] + X[:,:-1])**2 @ self.psi[:-1] / self.dim + np.sin(X) @ self.psi  / self.dim
        elif self.name == 'abs':
            z = X @ self.psi + (X[:,1:] + X[:,:-1])**2 @ self.psi[:-1] / self.dim + np.abs(X) @ self.psi  / self.dim
        elif self.name == 'sigmoid':
            z = X @ self.psi + (X[:,1:] + X[:,:-1])**2 @ self.psi[:-1] / self.dim + sigmoid(X) @ self.psi  / self.dim
        else:
            z = X @ self.psi + (X[:,1:] + X[:,:-1])**2 @ self.psi[:-1] / self.dim
        pi = scipy.special.expit( sc*(z+sh) )
        return pi
    
    def get_pi2y0(self, X, sc=1, sh=0.0):
        sh, sc = self.y_param
        if self.name == 'linear':
            z = X @ self.phi[:,0:1]
        elif self.name == 'toy':
            z = X @ self.phi[:,0:1] + (X[:,1:] + X[:,:-1])**2 @ self.phi[:-1,0:1] / self.dim
        elif self.name == 'sin':
            z = X @ self.phi[:,0:1] + (X[:,1:] + X[:,:-1])**2 @ self.phi[:-1,0:1] / self.dim
            z = z + np.sin(X) @ self.phi[:,0:1] / self.dim
        elif self.name == 'abs':
            z = X @ self.phi[:,0:1] + (X[:,1:] + X[:,:-1])**2 @ self.phi[:-1,0:1] / self.dim
            z = z + np.abs(X) @ self.phi[:,0:1] / self.dim
        elif self.name == 'sigmoid':
            z = X @ self.phi[:,0:1] + (X[:,1:] + X[:,:-1])**2 @ self.phi[:-1,0:1] / self.dim
            z = z + sigmoid(X) @ self.phi[:,0:1] / self.dim
        else:
            z = X @ self.phi[:,0:1] + (X[:,1:] + X[:,:-1])**2 @ self.phi[:-1,0:1] / self.dim
        pi = scipy.special.expit( sc*(z+sh) )
        return pi
    
    def get_pi2r(self, X, sc=1, sh=0.0):
        sh, sc = self.r_param
        if self.name == 'linear':
            print(f'This {self.name} Experiments.')
            z = X @ self.phi[:,1:2]
        elif self.name == 'toy':
            print(f'This {self.name} Experiments.')
            z = X @ self.phi[:,1:2] + (X[:,1:] + X[:,:-1])**2 @ self.phi[:-1,1:2] / self.dim
        elif self.name == 'sin':
            print(f'This {self.name} Experiments.')
            z = X @ self.phi[:,1:2] + (X[:,1:] + X[:,:-1])**2 @ self.phi[:-1,1:2] / self.dim
            z = z + np.sin(X) @ self.phi[:,1:2] / self.dim
        elif self.name == 'abs':
            print(f'This {self.name} Experiments.')
            z = X @ self.phi[:,1:2] + (X[:,1:] + X[:,:-1])**2 @ self.phi[:-1,1:2] / self.dim
            z = z + np.abs(X) @ self.phi[:,1:2] / self.dim
        elif self.name == 'sigmoid':
            print(f'This {self.name} Experiments.')
            z = X @ self.phi[:,1:2] + (X[:,1:] + X[:,:-1])**2 @ self.phi[:-1,1:2] / self.dim
            z = z + sigmoid(X) @ self.phi[:,1:2] / self.dim
        else:
            print(f'This {self.name} Experiments.')
            z = X @ self.phi[:,1:2] + (X[:,1:] + X[:,:-1])**2 @ self.phi[:-1,1:2] / self.dim
        pi = scipy.special.expit( sc*(z+sh) )
        return pi
    
    def get_pi2a(self, X, sc=1, sh=0.0):
        sh, sc = self.a_param
        if self.name == 'linear':
            z = X @ self.phi[:,2:3]
        elif self.name == 'toy':
            z = X @ self.phi[:,2:3] + (X[:,1:] + X[:,:-1])**2 @ self.phi[:-1,2:3] / self.dim
        elif self.name == 'sin':
            z = X @ self.phi[:,2:3] + (X[:,1:] + X[:,:-1])**2 @ self.phi[:-1,2:3] / self.dim
            z = z + np.sin(X) @ self.phi[:,2:3] / self.dim
        elif self.name == 'abs':
            z = X @ self.phi[:,2:3] + (X[:,1:] + X[:,:-1])**2 @ self.phi[:-1,2:3] / self.dim
            z = z + np.abs(X) @ self.phi[:,2:3] / self.dim
        elif self.name == 'sigmoid':
            z = X @ self.phi[:,2:3] + (X[:,1:] + X[:,:-1])**2 @ self.phi[:-1,2:3] / self.dim
            z = z + sigmoid(X) @ self.phi[:,2:3] / self.dim
        else:
            z = X @ self.phi[:,2:3] + (X[:,1:] + X[:,:-1])**2 @ self.phi[:-1,2:3] / self.dim

        pi = scipy.special.expit( sc*(z+sh) )
        return pi
    
    def shows(self, p):
        intervals = np.arange(0, 1.1, 0.1)
        counts = np.zeros(len(intervals) - 1)
        for i in range(len(intervals) - 1):
            counts[i] = np.sum((p >= intervals[i]) & (p < intervals[i + 1]))
        proportions = counts / len(p)
        cumulative_sums = np.cumsum(proportions)
        intervals_labels = [f"~{intervals[i+1]:.1f}" for i in range(len(intervals)-1)]
        df = pd.DataFrame({
            'P': proportions.round(4),
            'C': cumulative_sums.round(4),
        }).T
        df.columns = intervals_labels
        print(df)
    
    def generate(self, X, ifprint=True):
        C = self.get_cost(X)
        C1 = C - C + 0.4
        C2 = C - C + 0.5
        C3 = C - C + 0.6

        pi2t = self.get_pi2t(X)
        if ifprint: print('T-Param')
        if ifprint: self.shows(pi2t)
        T = self.rng.binomial(1, pi2t)

        pi2r = self.get_pi2r(X)
        if ifprint: print('R-Param')
        if ifprint: self.shows(pi2r)
        R = self.rng.binomial(1, pi2r)

        pi2y0 = self.get_pi2y0(X)
        if ifprint: print('Y-Param')
        if ifprint: self.shows(pi2y0)
        R2 = self.rng.binomial(1, pi2y0)

        if self.a_nonmn > 0:
            pi2a = self.get_pi2a(X)
            if ifprint: print('A-Param')
            if ifprint: self.shows(pi2a)
            A1 = self.rng.binomial(1, pi2a)
            Y0, Y1 = copy.deepcopy(R2), copy.deepcopy(R2)
            Y0[R==1] = 1-copy.deepcopy(A1[R==1])
            Y1[R==1] = copy.deepcopy(A1[R==1])
        else:
            Y1 = np.maximum(R, R2)
            Y0 = (R2 - R) == 1

        Y = Y1 * T + (1-T)*Y0


        if self.a_nonmn > 0:
            PY01 = pi2r*pi2a
            PY10 = pi2r*(1-pi2a)
        else:
            PY01 = pi2r
            PY10 = pi2r - pi2r
        PY00 = (1-pi2r)*(1-pi2y0)
        PY11 = (1-pi2r)*pi2y0

        print('P1: {}, P0: {}, PR=1: {}, tau: {}, P3: {}.'.format(np.mean((R)), np.mean((Y1-Y0)==0), np.mean((Y1-Y0)==1), np.mean(PY01), np.mean((Y1-Y0)==-1)))

        Py0  = PY10 + PY11
        Py1  = PY01 + PY11

        print("***"*10)
        self.shows(PY01)

        print(np.mean(Py1- Py0 ), np.mean(Y1- Y0 ))

        df = pd.DataFrame(np.concatenate([T, Y, Y0, Y1, C1, C2, C3, PY01, X], axis=1), columns=self.column)
        return df, myDataset(df)

args = set_args()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def _split_dataset(df):
  splits = args.ratio.strip().split('/')
  n_train,n_val,n_test = float(splits[0]),float(splits[1]),float(splits[2])
  n_units = len(df)
  feat_index = set(range(0, n_units))

  train_index = list(np.random.choice(list(feat_index),int(n_train * n_units),replace=False))
  val_index = list(np.random.choice(list(feat_index-set(train_index)),int(n_val * n_units),replace=False))
  test_index = list(feat_index-set(train_index)-set(val_index))

  train = df.iloc[train_index].reset_index(drop=True)
  val = df.iloc[val_index].reset_index(drop=True)
  test = df.iloc[test_index].reset_index(drop=True)
  return train, val, test
  
if __name__ == "__main__":
    name, exps, ratio, seed = args.name, args.exps, args.ratio, args.seed
    cvalue = args.cvalue
    data_name = f'{name}'

    for exp in range(1,1+exps):
        print(f'Data Generation to {data_name} {exp}.')
        data = DecisionMakingData(args, seed + 444*exp)
        data.set_config(dim=17)
        if 'Jobs' in data_name:
            path = './Data/Jobs/{}/'.format(exp-1)
            os.makedirs(path, exist_ok=True)

            df = pd.read_csv('./Jobs/Jobs{}.csv'.format(exp))

            X  = df.iloc[:,3:].values
            print("X-shape: ", X.shape)

            df, _ = data.generate(X)
            rng = np.random.default_rng(args.seed)
            train_df, val_df, test_df = _split_dataset(df)

            train_df.to_csv(path+'train.csv', index=False)
            val_df.to_csv(path+'val.csv', index=False)
            test_df.to_csv(path+'test.csv', index=False)

            