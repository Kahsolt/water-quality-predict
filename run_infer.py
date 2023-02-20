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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from modules.util import *
from modules.typing import *
import run as RT

WINDOW_TITLE  = 'Sequential Inference Demo'
WINDOW_SIZE   = (1000, 750)
HIST_FIG_SIZE = (8, 8)


def frame_left_pad(x:Frame, padlen:int) -> Frame:
  xlen = len(x)
  if xlen < padlen:
    x = np.pad(x, ((padlen - xlen, 0), (0, 0)), mode='edge')
  return x

def frame_shift(x:Frame, y:Frame) -> Frame:
  return np.concatenate([x[len(y):, :], y], axis=0)


class App:

  def __init__(self):
    self.setup_gui()

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
    RT.job = load_job(fp)
    name: str = RT.job_get('misc/name')
    assert name
    log_dp: Path = args.log_path / name
    assert log_dp.exists()
    RT.env['log_dp'] = log_dp
    seed_everything(RT.job_get('misc/seed'))

    # load job states
    @RT.require_model
    @RT.require_data
    def load_model_and_data():
      RT.env['model'] = RT.env['manager'].load(RT.env['model'], RT.env['log_dp'])
    load_model_and_data()

    seq = RT.env['seq']
    inlen = RT.job_get('dataset/in')
    print('  seq.shape:', seq.shape)
    seqlen = len(seq)
    res = max(seqlen // 100, inlen)
    tick = round(seqlen // 10 / 100) * 100

    self.sc_L.configure(to=seqlen, resolution=res, tickinterval=tick) ; self.sc_L_pack()
    self.sc_R.configure(to=seqlen, resolution=res, tickinterval=tick) ; self.sc_R_pack()

    self.var_L.set(tick)
    self.var_R.set(tick * 2)
    self.plot()

  def plot(self):
    if RT.job is None: return

    L = self.var_L.get()
    R = self.var_R.get()
    if L >= R: return

    seq: Seq     = RT.env['seq']
    stats: Stats = RT.env['stats']
    manager      = RT.env['manager']
    model: Model = RT.env['model']
    inlen: int   = RT.job_get('dataset/in')

    if 'predict with oracle (one step)':
      preds: List[Frame] = []
      loc = L
      while loc < R:
        x = seq[loc-inlen:loc, :]
        x = frame_left_pad(x, inlen)          # [I, D]
        y: Frame = manager.infer(model, x)    # [O, D]
        preds.append(y)
        loc += len(y)
      preds_o: Seq = np.concatenate(preds, axis=0)    # [T'=R-L+1, D]

    if 'predict with prediction (rolling)' and args.draw_rolling:
      preds: List[Frame] = []
      loc = L
      x = seq[loc-inlen:loc, :]
      x = frame_left_pad(x, inlen)            # [I, D]
      while loc < R:
        y: Frame = manager.infer(model, x)    # [O, D]
        preds.append(y)
        x = frame_shift(x, y)
        loc += len(y)
      preds_r: Seq = np.concatenate(preds, axis=0)    # [T'=R-L+1, D]

    if 'inv preprocess' and args.draw_rolling:
      namespace = globals()
      for (proc, st) in stats:
        invproc = namespace.get(f'{proc}_inv')
        if not invproc: continue
        print(f'  apply inv of {proc}')
        seq     = invproc(seq,   *st)
        preds_o = invproc(preds_o, *st)
        if args.draw_rolling:
          preds_r = invproc(preds_r, *st)

    if 'select range & channel':
      truth   = seq    [L:R,    0]
      preds_o = preds_o[:R-L+1, 0]      # [T'=R-L+1]
      if args.draw_rolling:
        preds_r = preds_r[:R-L+1, 0]    # [T'=R-L+1]

    self.ax.cla()
    self.ax.plot(truth,   'b', label='truth')
    self.ax.plot(preds_o, 'r', label='pred (oracle)')
    if args.draw_rolling:
      self.ax.plot(preds_r, 'g', label='pred (rolling)')
    self.fig.legend()
    self.fig.tight_layout()
    self.cvs.draw()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--log_path', default=Path('log'), type=Path, help='path to log root folder')
  parser.add_argument('--draw_rolling', action='store_true', help='whether draw rolling prediction')
  args = parser.parse_args()

  App()
