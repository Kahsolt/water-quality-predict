#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/14 

# 离散HMM
#   CategoricalHMM
#   MultinomialHMM
#   PoissonHMM
# 连续HMM
#   GaussianHMM
#   GMMHMM

import os
from argparse import ArgumentParser

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pylab as plt

np.random.seed(42)


''' Commandline '''
parser = ArgumentParser()
parser.add_argument('csv_fp', metavar='csv_fp', help='path to yoru .csv file')
parser.add_argument('--order', default=30, type=int, help='order number of HMM')
parser.add_argument('--n_iter', default=20, type=int, help='iter number of HMM')
parser.add_argument('--n_comp_min', default=20, type=int, help='min hidden state of HMM')
parser.add_argument('--n_comp_max', default=30, type=int, help='max hidden state of HMM')
parser.add_argument('--overwrite', action='store_true', help='force overwrite existing ckpt')
args = parser.parse_args()


''' Hparam '''
csv_fp = args.csv_fp
order = args.order
index = os.path.splitext(os.path.basename(csv_fp))[0]
save_fp = os.path.join('log', f'hmm_{index}_order={order}.pkl')
os.makedirs('log', exist_ok=True)


''' RData'''
rdata = pd.read_csv(csv_fp, parse_dates=['monitorTime'], index_col='monitorTime', date_parser=pd.to_datetime)
ts = rdata['monitorAvgValue']

x = ts.to_numpy()     # [T]


''' Preprocess'''
if 'smooth':
  y = np.empty_like(x)
  y[0] = x[0] ; y[-1] = x[-1]
  for i in range(1, len(y) - 1):
    m = (x[i-1] + x[i+1]) / 2
    if x[i] > m * 2 or x[i] < m / 2:
      y[i] = m
    else:
      y[i] = x[i]
  x = y

if not 'std':
  x = (x - x.mean()) / x.std()

if not 'qt':
  n_bins = 100
  x = ((x - x.min()) / (x.max() - x.min()) * n_bins).round()

if not 'draw':
  plt.plot(x, 'r')
  plt.show()


''' Data'''
X, Y = [], []
for i in range(len(x) - order):
  X.append([x[i + j] for j in range(order)])
  Y.append(x[i + order])
X_train = np.asarray(X)     # [T, order]
y_train = np.asarray(Y)     # [T]
X = X_train


''' Train '''
if not os.path.exists(save_fp) or args.overwrite:
  ''' Model '''
  models, scores = [ ],  [ ]
  for n_components in range(args.n_comp_min, args.n_comp_max+1, 5):
    for idx in range(10):                 # ten different random starting states
      try:
        model = hmm.GaussianHMM(n_components=n_components, random_state=idx, verbose=True, n_iter=args.n_iter)
        model.fit(X)
        score = model.score(X)    # log likely-hood: the larger the better

        models.append(model)
        scores.append(score)

        print(f'Converged: {model.monitor_.converged}, Score: {score}')
      except KeyboardInterrupt: exit(-1)
      except: pass
  model = models[np.argmax(scores)]   # get the best model
  print(f'The best model had a score of {max(scores)} and {model.n_components} components')

  ''' Save '''
  joblib.dump(model, save_fp)
else:
  ''' Load '''
  model = joblib.load(save_fp)


''' Plot '''
fig, ax = plt.subplots()
ax.imshow(model.transmat_)
ax.set_title('Transition Matrix')
ax.set_xlabel('State To')
ax.set_ylabel('State From')
ax.invert_yaxis()
plt.show()


''' Infer '''
# Evaluate this sequence
score1, probs = model.score_samples(X)   # [T, N=30]
pred = probs.argmax(axis=-1)
# use the Viterbi algorithm to predict the most likely sequence of states given the model
score2, states = model.decode(X)         # step by step infer

if not 'draw':
  plt.subplot(211) ; plt.plot(states) ; plt.title(score2)
  plt.subplot(212) ; plt.plot(pred)   ; plt.title(score1)
  cnt = sum(pred != states)
  # should they be equal
  if cnt > 0:
    plt.suptitle(f'found {cnt} points mismatch ({cnt / len(pred):%})')
  plt.show()


cp = int(len(X) * 0.9)
prev, post = X[:cp, :], X[cp:, :]   # [T1, N], [T2, N]
n_samples = len(post)
_, prev_state = model.decode(prev)   # [T1]
pred, pred_state = model.sample(n_samples, currstate=prev_state[-1])   # [T2, order], [T2]

truth_s = np.concatenate([prev[:, -1], post[:, -1]])    # only task last frame
pred_s  = np.concatenate([prev[:, -1], pred[:, -1]])

print('MAE:', mean_absolute_error(post[:, -1], pred[:, -1]))
print('MSE:', mean_squared_error (post[:, -1], pred[:, -1]))
print('R1:' , r2_score           (post[:, -1], pred[:, -1]))

if 'draw':
  plt.subplot(211) ; plt.plot(truth_s, 'b') ; plt.plot(pred_s, 'r')
  plt.subplot(212) ; plt.plot(states,  'b') ; plt.plot(np.concatenate([prev_state, pred_state]), 'r')
  plt.show()
