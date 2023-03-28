#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/20 

from argparse import ArgumentParser
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
import tkinter.filedialog as tkfdlg
from traceback import print_exc

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from modules.util import *
from modules.typing import *

from run import *
import matplotlib ; matplotlib.use('QtAgg')

WINDOW_TITLE  = 'Sequential Inference Demo'
WINDOW_SIZE   = (1000, 750)
HIST_FIG_SIZE = (8, 8)


class App:

  def __init__(self):
    self.setup_gui()

    self.env: Env = None

    try:
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

    # Top: control
    frm1 = ttk.Frame(wnd)
    frm1.pack(side=tk.TOP, anchor=tk.N, expand=tk.YES, fill=tk.X)
    if True:
      self.var_job_file = tk.StringVar(frm1)
      self.var_L = tk.IntVar(frm1, value=0)
      self.var_R = tk.IntVar(frm1, value=0)

      frm11 = ttk.LabelFrame(frm1, text='Job File')
      frm11.pack(expand=tk.YES, fill=tk.X)
      if True:
        ent = ttk.Entry(frm11, textvariable=self.var_job_file, state=tk.DISABLED)
        ent.pack(side=tk.LEFT, anchor=tk.W, expand=tk.YES, fill=tk.X)

        lb = ttk.Button(frm11, text='Open..', command=self.open)
        lb.pack(side=tk.RIGHT, anchor=tk.E)

      frm12 = ttk.LabelFrame(frm1, text='Predict Range')
      frm12.pack(expand=tk.YES, fill=tk.X)
      if True:
        self.sc_L = tk.Scale(frm12, variable=self.var_L, command=lambda _: self.plot(), from_=0, to=10, orient=tk.HORIZONTAL)
        self.sc_L_pack = lambda: self.sc_L.pack(side=tk.LEFT, anchor=tk.W, expand=tk.YES, fill=tk.X)
        self.sc_L_pack()

        self.sc_R = tk.Scale(frm12, variable=self.var_R, command=lambda _: self.plot(), from_=0, to=10, orient=tk.HORIZONTAL)
        self.sc_R_pack = lambda: self.sc_R.pack(side=tk.RIGHT, anchor=tk.E, expand=tk.YES, fill=tk.X)
        self.sc_R_pack()

    # bottom: plot
    frm2 = ttk.Frame(wnd)
    frm2.pack(side=tk.BOTTOM, expand=tk.YES, fill=tk.BOTH)
    if True:
      fig, ax = plt.subplots()
      fig.set_size_inches(HIST_FIG_SIZE)
      fig.tight_layout()
      cvs = FigureCanvasTkAgg(fig, frm2)
      cvs.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)

      self.fig, self.ax, self.cvs = fig, ax, cvs

  def open(self):
    fp = tkfdlg.askopenfilename(title='Open a job.yaml file..', filetypes=[('yaml files', '*.yaml')])
    if not fp: return
    fp = Path(fp)
    if not fp.exists():
      tkmsg.showerror('Error', 'File not exists!')
      return
    self.var_job_file.set(fp)

    # init job
    self.env = load_env(fp)
    env: Env = self.env
    job: Descriptor = env['job']

    self.is_model_arima = 'ARIMA' in job['model/name']
    self.is_task_rgr = env['manager'].TASK_TYPE == TaskType.RGR
    print(f'  is_task_rgr: {self.is_task_rgr}')

    # precalc whole seq
    seq:   Seq   = env['seq']     # transformed
    label: Seq   = env['label']
    stats: Stats = env['stats']

    self.preds_o: Seq = predict_with_oracle(env)
    self.preds_o = np.pad(self.preds_o, (len(seq) - len(self.preds_o), 0), mode='edge')
    if args.draw_rolling:
      self.preds_r: Seq = predict_with_predicted(env)
      self.preds_r = np.pad(self.preds_o, (len(seq) - len(self.preds_r), 0), mode='edge')
    if self.is_task_rgr:
      self.truth = inv_transforms(seq, stats)
    else:
      self.truth = label

    # set view range
    print(f'  seq.shape: {seq.shape}')
    seqlen = len(seq)
    inlen: int = job.get('dataset/inlen', 72)
    res = max(seqlen // 100, inlen)
    tick = min(10, round(seqlen // 10 / 100) * 100)

    self.sc_L.configure(to=seqlen, resolution=res, tickinterval=tick) ; self.sc_L_pack()
    self.sc_R.configure(to=seqlen, resolution=res, tickinterval=tick) ; self.sc_R_pack()
    self.var_L.set(tick)
    self.var_R.set(tick * 2)

    self.plot()

  def plot(self):
    if self.env is None: return

    L = self.var_L.get()
    R = self.var_R.get()
    if L >= R: return

    if 'select range & channel':
      truth   = self.truth  [L:R, 0]
      preds_o = self.preds_o[L:R, 0]      # [T'=R-L+1]
      if args.draw_rolling:
        preds_r = self.preds_r[L:R, 0]    # [T'=R-L+1]

    if 'show acc':
      if self.is_task_rgr:
        mae = np.abs(truth - preds_o).mean()
        print(f'>> mae: {mae}')
      else:
        acc = (truth == preds_o).sum() / len(truth)
        print(f'>> acc: {acc:.3%}')

    self.ax.cla()
    self.ax.plot(truth,   'b', label='truth')
    self.ax.plot(preds_o, 'r', label='pred (oracle)')
    if args.draw_rolling:
      self.ax.plot(preds_r, 'g', label='pred (rolling)')
    if not self.is_task_rgr:
      self.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    self.fig.legend()
    self.fig.tight_layout()
    self.cvs.draw()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--draw_rolling', action='store_true', help='whether draw rolling prediction')
  args = parser.parse_args()

  App()
