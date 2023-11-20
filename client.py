#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/09

# GUI client for server

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
from traceback import print_exc
from typing import Dict, Any
from argparse import ArgumentParser

import numpy as np
import requests as R
from requests import Response

from modules.utils import ts_now, ndarray_to_list

WINDOW_TITLE = 'Inference Client'
WINDOW_SIZE  = (500, 500)
CB_WIDTH     = 200
TX_LINES_IN  = 12
TX_LINES_OUT = 6

HTTP_FAIL = object()


def GET(api:str) -> Dict:
  url = EP(api)
  print(f'[GET] {url}')
  resp: Response = R.get(url)
  if not resp.ok:
    tkmsg.showerror('Error', vars(resp))
    return HTTP_FAIL
  r = resp.json()
  if not r['ok']:
    tkmsg.showerror('Error', r)
    return HTTP_FAIL
  return r['data']

def POST(api:str, payload:Any) -> Dict:
  url = EP(api)
  print(f'[POST] {url}')
  resp: Response = R.post(url, json=payload, timeout=5)
  if not resp.ok:
    tkmsg.showerror('Error', vars(resp))
    return HTTP_FAIL
  r = resp.json()
  if not r['ok']:
    tkmsg.showerror('Error', r)
    return HTTP_FAIL
  return r['data']


warn_data_too_short = False
warn_data_too_long  = False

class App:

  def __init__(self):
    self.setup_gui()

    self.tasks: Dict[str, Dict[str, Dict[str, Any]]] = {}     # {'task': {'job': info}}
    self.cur_task: str = None
    self.cur_job: str = None

    try:
      self.setup_workspace()
      self.wnd.mainloop()
    except KeyboardInterrupt:
      self.wnd.destroy()
    except: print_exc()

  def setup_gui(self):
    # window
    wnd = tk.Tk()
    W, H = wnd.winfo_screenwidth(), wnd.winfo_screenheight()
    w, h = WINDOW_SIZE
    wnd.geometry(f'{w}x{h}+{(W-w)//2}+{(H-h)//2}')
    wnd.resizable(False, False)
    wnd.title(WINDOW_TITLE)
    wnd.protocol('WM_DELETE_WINDOW', wnd.quit)
    self.wnd = wnd

    # Top: Model
    frm1 = ttk.LabelFrame(wnd, text='Model')
    frm1.pack(side=tk.TOP, anchor=tk.N)
    if True:
      self.var_task = tk.StringVar(wnd)
      self.var_job  = tk.StringVar(wnd)

      # select task/job
      frm11 = ttk.Frame(frm1)
      frm11.pack()
      if True:
        lb = tk.Label(frm11, text='Task:')
        lb.pack(side=tk.LEFT, padx=2)
        cb = ttk.Combobox(frm11, values=[], state='readonly', textvariable=self.var_task, width=CB_WIDTH)
        cb.bind('<<ComboboxSelected>>', lambda evt: self._chg_task())
        cb.pack()
        self.cb_task = cb

      frm12 = ttk.Frame(frm1)
      frm12.pack()
      if True:
        lb = tk.Label(frm12, text='Job:')
        lb.pack(side=tk.LEFT, padx=5)
        cb = ttk.Combobox(frm12, values=[], state='readonly', textvariable=self.var_job, width=CB_WIDTH)
        cb.bind('<<ComboboxSelected>>', lambda evt: self._chg_job())
        cb.pack()
        self.cb_job = cb

      # model info
      frm13 = ttk.Frame(frm1)
      frm13.pack()
      if True:
        self.var_model_info = tk.StringVar(wnd, '>> No job selected...')

        lb = tk.Label(frm13, textvariable=self.var_model_info, fg='blue')
        lb.pack(expand=tk.YES, fill=tk.BOTH, padx=20)

    # middle: input / output
    frm2 = ttk.Frame(wnd)
    frm2.pack(expand=tk.YES, fill=tk.BOTH)
    if True:
      frm21 = ttk.LabelFrame(frm2, text='Input')
      frm21.pack(expand=tk.YES, fill=tk.BOTH)
      if True:
        tx = tk.Text(frm21, height=TX_LINES_IN)
        tx.pack(expand=tk.YES, fill=tk.BOTH)
        self.tx_input = tx

      frm22 = ttk.LabelFrame(frm2, text='Output')
      frm22.pack(expand=tk.YES, fill=tk.BOTH)
      if True:
        tx = tk.Text(frm22, height=TX_LINES_OUT)
        tx.pack(expand=tk.YES, fill=tk.BOTH)
        self.tx_output = tx

    # bottom: buttons
    frm3 = ttk.Frame(wnd)
    frm3.pack(side=tk.BOTTOM, expand=tk.YES, fill=tk.BOTH)
    if True:
      btn = tk.Button(text='Random', fg='blue', command=self.rand_input)
      btn.pack(side=tk.LEFT, expand=tk.YES, fill=tk.X, padx=16)
      btn = tk.Button(text='Query!', fg='red', command=self.query)
      btn.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X, padx=16)

      frm31 = ttk.Frame(wnd)
      frm31.pack(side=tk.BOTTOM, expand=tk.YES, fill=tk.BOTH)
      if True:
        self.var_rand_w = tk.DoubleVar(wnd, 1.0)
        lb = tk.Label(frm31, text='Random Weight:')
        lb.pack()
        ent = tk.Entry(frm31, textvariable=self.var_rand_w)
        ent.pack()

  def setup_workspace(self):
    data = GET('/task')
    if data is HTTP_FAIL: return

    self.tasks.clear()
    tasks = sorted(data['tasks'])
    for task in tasks:
      self.tasks[task] = {}
    self.cb_task.config(values=tasks)
    self.cur_task = None
    self.cur_job = None

  def _chg_task(self):
    task = self.cb_task.get()
    if task == self.cur_task: return
    self.cur_task = task
    if task not in self.tasks: return

    if not len(self.tasks[task]):
      data = GET(f'/task/{task}')
      if data is HTTP_FAIL: return

      self.tasks[task] = data['jobs']

    jobs = sorted(self.tasks[task].keys())
    self.cb_job.config(values=jobs)
    self.var_job.set('')
    self.cur_job = None
  
    self.refresh_model_info()

  def _chg_job(self):
    job = self.cb_job.get()
    if job == self.cur_job: return
    self.cur_job = job

    self.refresh_model_info()

  def refresh_model_info(self):
    task = self.var_task.get()
    job  = self.var_job .get()
    if not task or not job: return

    info = self.tasks[task][job]
    try:
      inlen = info['inlen']
      metric: Dict[str, float] = info['scores']
      if info['type'] == 'clf':
        metric_str = {k: f'{v:.3%}' for k, v in metric.items()}
      else:
        metric_str = {k: f'{v:.5f}' for k, v in metric.items()}
      summary = '\n'.join([
        f'inlen: {inlen}',
        f'metric: {metric_str}',
      ])
    except:
      summary = info
    self.var_model_info.set(summary)

  def rand_input(self):
    task = self.cur_task
    job = self.cur_job
    if not task or not job: return tkmsg.showerror('Error', 'no job selected')
    w = self.var_rand_w.get()

    inlen = self.tasks[task][job]['inlen']
    x = np.random.uniform(low=0.0, high=1.0, size=[inlen]).astype(np.float32) * w
    x = [round(e, 4) for e in ndarray_to_list(x)]

    self.tx_input.delete('0.0', tk.END)
    self.tx_input.insert('0.0', str(x))

  def query(self):
    task = self.cur_task
    job = self.cur_job
    if not task or not job: return tkmsg.showerror('Error', 'no job selected')
  
    x_str = self.tx_input.get('0.0', tk.END).strip()
    if not x_str: x_str = '[]'
    try: x = eval(x_str)
    except: return tkmsg.showerror('Error', f'cannot eval input string: {x_str}')
    if type(x) != list: return tkmsg.showerror('Error', 'input should be a python list')
    for e in x:
      if not isinstance(e, (float, int)):
        return tkmsg.showerror('Error', f'list items should be float|int numbers, but found: {type(e)!r} {e}')

    inlen = self.tasks[task][job]['inlen']
    xlen = len(x)
    if xlen < inlen:
      global warn_data_too_short
      if not warn_data_too_short:
        warn_data_too_short = True
        tkmsg.showwarning('Warn', 'input length < required inlen, will zero-pad the left part')
      x = [1e-8] * (inlen - xlen) + x
    if xlen > inlen:
      global warn_data_too_long
      if not warn_data_too_long:
        warn_data_too_long = True
        tkmsg.showwarning('Warn', 'input length > required inlen, will truncate the left part')
      x = x[:-inlen]

    self.tx_output.delete('0.0', tk.END)

    x = [[e] for e in x]    # [N] => [N, 1]
    ts = ts_now()
    t = [ts + i * 3600 for i in range(len(x))]
    payload = { 'data': x, 'time': t }
    data = POST(f'/infer/{task}/{job}', payload)
    if data is HTTP_FAIL: return

    self.tx_output.insert('0.0', str(data['pred']))


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-H', '--host', type=str, default='0.0.0.0')
  parser.add_argument('-P', '--port', type=int, default=5000)
  args = parser.parse_args()

  API_BASE = f'http://{args.host}:{args.port}'
  EP = lambda api: f'{API_BASE}{api}'

  try:
    App()
  except KeyboardInterrupt:
    print('Exit by Ctrl+C')
  except:
    print_exc()
