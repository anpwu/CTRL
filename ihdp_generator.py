import os
import numpy as np
import pandas as pd
import argparse
import scipy.special
from utils import myDataset, torchDataset, set_args
from src.preprocess import run as runReg

np.set_printoptions(suppress=True, precision=6)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class DecisionMakingData():
    def __init__(self, args, seed=2024):
        self.name = args.name
        self.cvalue = args.cvalue
        self.vvalue = args.vvalue
        self.t_param = args.t_param
        self.r_param = args.r_param
        self.y_param = args.y_param
        self.c_param = args.c_param
        self.v_param = args.v_param
        self.rng  = np.random.default_rng(seed)
        self.set_config()
        
    def set_config(self, dim=3):
        self.cov = 0.3
        self.dim = dim
        self.column = ['X' + str(i) for i in range(self.dim)] 
        self.column = self.column + ['C','S','V','W','T','Y','G0','G1','PT','PR','PR2','Q0','Q1','Q01','Q00','Q11']
        self.set_params()
        self.get_covariance()
    
    def set_params(self):
        self.psi = self.rng.uniform(-1.0, 1.0, size=(self.dim, 1)).round(1)
        self.csi = self.rng.uniform(-1.0, 1.0, size=(self.dim, 1)).round(1)
        self.vsi = self.rng.uniform(-1.0, 1.0, size=(self.dim, 1)).round(1)
        self.phi = self.rng.uniform(-1.0, 1.0, size=(self.dim, 2)).round(1)

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
    
    def generate(self, num=2000, ifprint=True):
        X = self.get_x(num)
        
        V = self.get_income(X)

        C = self.get_cost(X)
        if ifprint: self.shows(C)

        pi2t = self.get_pi2t(X)
        if ifprint: self.shows(pi2t)
        T = self.rng.binomial(1, pi2t)

        pi2r = self.get_pi2r(X)
        if ifprint: self.shows(pi2r)
        R = self.rng.binomial(1, pi2r)

        pi2y0 = self.get_pi2y0(X)
        if ifprint: self.shows(pi2y0)
        R2 = self.rng.binomial(1, pi2y0)
        Y1 = np.maximum(R, R2)
        Y0 = (R2 - R) == 1

        Y = Y1 * T + (1-T)*Y0

        PY01 = pi2r
        PY00 = (1-pi2r)*(1-pi2y0)
        PY11 = (1-pi2r)*pi2y0

        Py0  = (1-pi2r)*pi2y0
        Py1  = pi2r + (1-pi2r)*pi2y0

        df = pd.DataFrame(np.concatenate([X, C, C, V, V, T, Y, Y0, Y1, pi2t, pi2r, pi2y0, Py0, Py1, PY01, PY00,PY11], axis=1), columns=self.column)
        return df, myDataset(df)

    def testing(self, num=500, rep=40, ifprint=True):
        X_ = self.get_x(num)
        X = np.tile(X_, (rep, 1))
        V = self.get_income(X)
        C = self.get_cost(X)
    
        pi2t = self.get_pi2t(X)
        if ifprint: self.shows(pi2t)
        T = self.rng.binomial(1, pi2t)
        
        pi2r = self.get_pi2r(X)
        if ifprint: self.shows(pi2r)
        R = self.rng.binomial(1, pi2r)

        pi2y0 = self.get_pi2y0(X)
        if ifprint: self.shows(pi2y0)
        R2 = self.rng.binomial(1, pi2y0) 
        Y1 = np.maximum(R, R2) 
        Y0 = (R2 - R) == 1

        Y = Y1 * T + (1-T)*Y0

        PY01 = pi2r
        PY00 = (1-pi2r)*(1-pi2y0)
        PY11 = (1-pi2r)*pi2y0

        Py0  = (1-pi2r)*pi2y0
        Py1  = pi2r + (1-pi2r)*pi2y0

        df = pd.DataFrame(np.concatenate([X, C, C, V, V, T, Y, Y0, Y1, pi2t, pi2r, pi2y0, Py0, Py1, PY01, PY00,PY11], axis=1), columns=self.column)
        return df, myDataset(df)

    def get_pi2t(self, X):
        sh, sc = self.t_param
        if self.name == 'linear':
            z = X @ self.psi
        elif self.name == 'toy':
            z = X @ self.psi + X[:,1:] * X[:,:-1] @ self.psi[:-1]
        elif self.name == 'sin':
            z = np.sin(X) @ self.psi + X[:,1:] * X[:,:-1] @ self.psi[:-1]
        elif self.name == 'poly':
            z = 0.1*(X**2) @ self.psi + X[:,1:] * X[:,:-1] @ self.psi[:-1]
        elif self.name == 'sigmoid':
            z = 0.1*sigmoid(X) @ self.psi + X[:,1:] * X[:,:-1] @ self.psi[:-1]
        else:
            z = X @ self.psi + X[:,1:] * X[:,:-1] @ self.psi[:-1]
        pi = scipy.special.expit( sc*(z+sh) )
        return pi
    
    def get_pi2y0(self, X, sc=1, sh=0.0):
        sh, sc = self.y_param
        if self.name == 'linear':
            z = X @ self.phi[:,0:1]
        elif self.name == 'toy':
            z = X @ self.phi[:,0:1] + X[:,1:] * X[:,:-1] @ self.phi[:-1,0:1]
        elif self.name == 'sin':
            z = np.sin(X) @ self.phi[:,0:1] + X[:,1:] * X[:,:-1] @ self.phi[:-1,0:1]
        elif self.name == 'poly':
            z = 0.1*(X**2) @ self.phi[:,0:1] + X[:,1:] * X[:,:-1] @ self.phi[:-1,0:1]
        elif self.name == 'sigmoid':
            z = 0.1*sigmoid(X) @ self.phi[:,0:1] + X[:,1:] * X[:,:-1] @ self.phi[:-1,0:1]
        else:
            z = X @ self.phi[:,0:1] + X[:,1:] * X[:,:-1] @ self.phi[:-1,0:1]
        pi = scipy.special.expit( sc*(z+sh) )
        return pi
    
    def get_pi2r(self, X, sc=1, sh=0.0):
        sh, sc = self.r_param
        if self.name == 'linear':
            z = X @ self.phi[:,1:2]
        elif self.name == 'toy':
            z = X @ self.phi[:,1:2] + X[:,1:] * X[:,:-1] @ self.phi[:-1,1:2]
        elif self.name == 'sin':
            z = np.sin(X) @ self.phi[:,1:2] + X[:,1:] * X[:,:-1] @ self.phi[:-1,1:2]
        elif self.name == 'poly':
            z = 0.1 * (X**2) @ self.phi[:,1:2] + X[:,1:] * X[:,:-1] @ self.phi[:-1,1:2]
        elif self.name == 'sigmoid':
            z = 0.1 * sigmoid(X) @ self.phi[:,1:2] + X[:,1:] * X[:,:-1] @ self.phi[:-1,1:2]
        else:
            z = X @ self.phi[:,1:2] + X[:,1:] * X[:,:-1] @ self.phi[:-1,1:2]
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

def shows(p):
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


args = set_args()

def _split_dataset(df):
  df = df.sample(frac = 1)
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

    for exp in range(exps):
        print(f'Data Generation to {data_name} {exp}.')
        if 'IHDP' in data_name:
            df = pd.read_csv('./IHDP/IHDP1.csv')

            rng = np.random.default_rng(args.seed)
            train_data, val_data, test_data = _split_dataset(df)
            train_x = train_data.iloc[:,5:].values
            val_x = val_data.iloc[:,5:].values
            test_x = test_data.iloc[:,5:].values

            train_t = train_data['treatment'].values
            val_t = val_data['treatment'].values
            test_t = test_data['treatment'].values

            train_u0 = train_data['mu0'].values
            val_u0 = val_data['mu0'].values
            test_u0 = test_data['mu0'].values            

            train_u1 = train_data['mu1'].values
            val_u1 = val_data['mu1'].values
            test_u1 = test_data['mu1'].values      

            all_u1 = np.concatenate([train_u1, val_u1, test_u1])  
            all_u0 = np.concatenate([train_u0, val_u0, test_u0])
            print(f'mean: {np.mean(all_u0):.4f}, {np.mean(all_u1):.4f}')
            print(f'std: {np.std(all_u0):.4f}, {np.std(all_u1):.4f}')

            mode, y0_mean, y1_mean, y0_std, y1_std = args.mode, args.y0_mean, args.y1_mean, args.y0_std, args.y1_std
            path = './Data/IHDP_{}_{}_{}_{}_{}/{}/'.format(mode, y0_mean, y1_mean, y0_std, y1_std, exp)
            
            if y0_mean <= 0:
                y0_mean = np.mean(all_u0)
            if y1_mean <= 0:
                y1_mean = np.mean(all_u1)
            if y0_std <= 0:
                y0_std = 1/np.std(all_u0)
            if y1_std <= 0:
                y1_std = 1/np.std(all_u1)

            os.makedirs(path, exist_ok=True)

            if mode == 1:
                train_p0 = sigmoid((train_u0-y0_mean)*y0_std)
                train_p1 = sigmoid((train_u1-y1_mean)*y1_std)
                train_R = rng.binomial(1, train_p1)
                train_R2 = rng.binomial(1, train_p0)
                train_y0 = (train_R2 - train_R) == 1
                train_y1 = np.maximum(train_R, train_R2)
                train_P = train_p1
                train_y = train_y0 * (1 - train_t) + train_y1 * train_t
                print('train R=1 {}'.format(exp), sum(train_y1 > train_y0) / len(train_y0))
                print('train_p0')
                shows(train_p0)
                print('train_p1')
                shows(train_p1)
                print('train_P')
                shows(train_P)

                val_p0 = sigmoid((val_u0-y0_mean)*y0_std)
                val_p1 = sigmoid((val_u1-y1_mean)*y1_std)
                val_R = rng.binomial(1, val_p1)
                val_R2 = rng.binomial(1, val_p0)
                val_y0 = (val_R2 - val_R) == 1
                val_y1 = np.maximum(val_R, val_R2)
                val_P = val_p1
                val_y = val_y0 * (1 - val_t) + val_y1 * val_t
                print('val R=1 {}'.format(exp), sum(val_y1 > val_y0) / len(val_y0))

                test_p0 = sigmoid((test_u0-y0_mean)*y0_std)
                test_p1 = sigmoid((test_u1-y1_mean)*y1_std)
                test_R = rng.binomial(1, test_p1)
                test_R2 = rng.binomial(1, test_p0)
                test_y0 = (test_R2 - test_R) == 1
                test_y1 = np.maximum(test_R, test_R2)
                test_P = test_p1
                test_y = test_y0 * (1 - test_t) + test_y1 * test_t
                print('test R=1 {}'.format(exp), sum(test_y1 > test_y0) / len(test_y0))
            else:
                # 𝑌(+1)≥𝑌(−1)
                train_p0 = sigmoid((train_u0-y0_mean)*y0_std)
                train_p1 = sigmoid((train_u1-y1_mean)*y1_std)
                train_y0 = rng.binomial(1, train_p0)
                train_y1 = rng.binomial(1, train_p1)
                train_P = (1-train_p0)*train_p1
                delete_list_train = np.where(train_y0 > train_y1)[0]
                print(f'train_P: {np.mean(train_P):.4f}')
                print('delete_list_train', len(delete_list_train))
                train_y1[delete_list_train] = 1
                train_y = train_y0 * (1 - train_t) + train_y1 * train_t
                print('train R=1 {}'.format(exp), sum(train_y1 > train_y0) / len(train_y0))
                print('train_p0')
                shows(train_p0)
                print('train_p1')
                shows(train_p1)
                print('train_P')
                shows(train_P)

                val_y0 = rng.binomial(1, sigmoid((val_u0-y0_mean)*y0_std))
                val_y1 = rng.binomial(1, sigmoid((val_u1-y1_mean)*y1_std))
                val_P = (1-sigmoid((val_u0-y0_mean)*y0_std))*sigmoid((val_u1-y1_mean)*y1_std)
                delete_list_val = np.where(val_y0 > val_y1)[0]
                val_y1[delete_list_val] = 1
                val_y = val_y0 * (1 - val_t) + val_y1 * val_t
                print('val R=1 {}'.format(exp), sum(val_y1 > val_y0) / len(val_y0))

                test_y0 = rng.binomial(1, sigmoid((test_u0-y0_mean)*y0_std))
                test_y1 = rng.binomial(1, sigmoid((test_u1-y1_mean)*y1_std))
                test_P = (1-sigmoid((test_u0-y0_mean)*y0_std))*sigmoid((test_u1-y1_mean)*y1_std)
                delete_list_test = np.where(test_y0 > test_y1)[0]
                test_y1[delete_list_test] = 1
                test_y = test_y0 * (1 - test_t) + test_y1 * test_t
                print('test R=1 {}'.format(exp), sum(test_y1 > test_y0) / len(test_y0), test_t.shape)
            
            cvalue_train1 = 0.4*np.ones(len(train_t))
            cvalue_val1 = 0.4*np.ones(len(val_t))
            cvalue_test1 = 0.4*np.ones(len(test_t))

            cvalue_train2 = 0.5*np.ones(len(train_t))
            cvalue_val2 = 0.5*np.ones(len(val_t))
            cvalue_test2 = 0.5*np.ones(len(test_t))

            cvalue_train3 = 0.6*np.ones(len(train_t))
            cvalue_val3 = 0.6*np.ones(len(val_t))
            cvalue_test3 = 0.6*np.ones(len(test_t))                    

            train_df = pd.DataFrame(np.c_[train_t,train_y,train_y0,train_y1,cvalue_train1,cvalue_train2,cvalue_train3,train_P,train_x], columns=['t','y','u0','u1','c1','c2','c3','P']+[f'x{i}' for i in range(train_x.shape[1])])
            val_df = pd.DataFrame(np.c_[val_t,val_y,val_y0,val_y1,cvalue_val1,cvalue_val2,cvalue_val3,val_P,val_x], columns=['t','y','u0','u1','c1','c2','c3','P']+[f'x{i}' for i in range(val_x.shape[1])])
            test_df = pd.DataFrame(np.c_[test_t,test_y,test_y0,test_y1,cvalue_test1,cvalue_test2,cvalue_test3,test_P,test_x], columns=['t','y','u0','u1','c1','c2','c3','P']+[f'x{i}' for i in range(test_x.shape[1])])

            os.makedirs(path, exist_ok=True)
            train_df.to_csv(path+'train.csv', index=False)
            val_df.to_csv(path+'val.csv', index=False)
            test_df.to_csv(path+'test.csv', index=False)            
