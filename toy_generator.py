import os
import numpy as np
import pandas as pd
import copy
import argparse
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
        self.column = ['X' + str(i) for i in range(self.dim)] 
        self.column = self.column + ['C','S','V','W','T','Y','G0','G1','PT','PR','PR2','Q0','Q1','Q01','Q00','Q11']
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
    
    def generate(self, num=2000, ifprint=True):
        X = self.get_x(num)
        
        V = self.get_income(X)

        C = self.get_cost(X)

        pi2t = self.get_pi2t(X)
        T = self.rng.binomial(1, pi2t)

        pi2r = self.get_pi2r(X)
        if ifprint: print('R-Param')
        if ifprint: self.shows(pi2r)
        R = self.rng.binomial(1, pi2r)

        pi2y0 = self.get_pi2y0(X)
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

        df = pd.DataFrame(np.concatenate([X, C, C, V, V, T, Y, Y0, Y1, pi2t, PY01, pi2y0, Py0, Py1, PY01, PY00,PY11], axis=1), columns=self.column)
        return df, myDataset(df), np.mean((Y1-Y0)==-1)

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
        return df, myDataset(df), np.mean((Y1-Y0)==-1)

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

args = set_args()

if __name__ == "__main__":
    name, exps, dim, num, tnum, trep, seed = args.name, args.exps, args.dim, args.num+args.vnum, args.tnum, args.trep, args.seed
    cvalue, vvalue = args.cvalue, args.vvalue
    t0, t1 = args.t_param
    r0, r1 = args.r_param
    a0, a1 = args.a_param
    y0, y1 = args.y_param
    nonmoni = args.a_nonmn
    dataPath  = args.path
    data_name = f'{name}_{dim}_{num}_{tnum}_{trep}_{cvalue}_{vvalue}'
    data_name = data_name + f'/{nonmoni}_{r0}_{r1}_{a0}_{a0}'

    if not os.path.exists(f'{dataPath}/{data_name}/{exps-1}/') or args.again:
        data = DecisionMakingData(args, seed)
        data.set_config(dim)
        for exp in range(exps):
            data_path = f'{dataPath}/{data_name}/{exp}/'
            print(f'Data Generation to {data_path}.')
            
            train_df, _, apro = data.generate(num, args.ifprint)
            test_df,  _, _  = data.testing(tnum, trep, False)

            train_np = myDataset(train_df)
            test_np  = myDataset(test_df)
            train    = torchDataset(train_np)
            test     = torchDataset(test_np)

            if args.cvalue <= 0.0:
                train_S, test_S = runReg(exp, args, train, _, test, 'S')

                train_df['S'] = train_S.detach().cpu().numpy()
                test_df['S'] = test_S.detach().cpu().numpy() 

            if args.vvalue <= 0.0:
                train_W, test_W = runReg(exp, args, train, _, test, 'W')

                train_df['W'] = train_W.detach().cpu().numpy()
                test_df['W'] = test_W.detach().cpu().numpy()  
            
            os.makedirs(data_path, exist_ok=True)
            train_df.to_csv(data_path+f'train.csv', index=False)
            test_df.to_csv(data_path+'test.csv', index=False)
            np.savetxt(data_path+'parameters.csv', np.concatenate([data.psi, data.phi],1).T, fmt="%.2f", delimiter=",")
