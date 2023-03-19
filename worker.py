#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/19 

from pathlib import Path
from queue import Queue, Empty
from pprint import pformat
from threading import Thread, Event

from modules.descriptor import *
from modules.dataset import *
from modules.util import *
from modules.typing import *

from config import *


class Trainer:

  def __init__(self):
    self.envs: Dict[str, Env] = { }
    self.queue = Queue()
    self.evt = Event()
    self.worker = Thread(target=worker, args=(self.evt,self.queue))

    self.run_meta: List[Run] = load_json(LOG_PATH / TASK_FILE, [])
    self._resume()

  def _resume(self):
    for run in self.run_meta:
      if Status(run['status']) in [Status.QUEUING, Status.CREATED, Status.RUNNING]:
        run['status'] = Status.QUEUING.value
        run['ts_update'] = ts_now()
        self.queue.put((run, self))

  def save_meta(self):
    save_json(LOG_PATH / TASK_FILE, self.run_meta)

  def add_task(self, name:str, init_fp:Path):
    run = new_runtime_entry()   # Status.QUEUING
    run['id'] = len(self.run_meta) + 1
    run['name'] = name
    run['task_init_pack'] = init_fp
    self.run_meta.append(run)
    self.queue.put((run, self))

  def start(self):
    self.save_meta()
    self.worker.start()

  def stop(self):
    self.evt.set()
    self.worker.join()
    self.save_meta()


def worker(evt:Event, queue:Queue):
  while not evt.is_set():
    payload = None
    while payload is None:
      try: payload: Tuple[Run, Trainer] = queue.get(timeout=CHECK_TASK_EVERY)
      except Empty: pass
      if evt.is_set(): return

    run, runtime = payload

    if run['status'] == Status.QUEUING.value:
      try: 
        task_create(run, runtime)
        run['status'] = Status.CREATED.value
      except:
        run['status'] = Status.FAILED.value

    if run['status'] == Status.CREATED.value:
      run['status'] = Status.RUNNING.value

    if run['status'] == Status.RUNNING.value:
      try:
        task_run(run, runtime)
        run['status'] = Status.FINISHED.value
      except:
        run['status'] = Status.FAILED.value

    if run['status'] == Status.FINISHED.value:
      task_cleanup(run, runtime)

    runtime.save_meta()
    queue.task_done()


def task_create(run:Run, runtime:Trainer):
  # unpack task_init_pack
  run['info'] = 'setup task log folder'
  runtime.save_meta()

  init_fp = Path(run['task_init_pack'])
  init: TaskInit = load_pickle(init_fp)

  name = init['name'] ; assert name == run['name']
  target = init['target'] or ['all']
  jobs = init['jobs'] or [job.stem for job in JOB_PATH.iterdir() if job.suffix == '.yaml']

  log_dp: Path = LOG_FILE / name
  log_dp.mkdir(exist_ok=True, parents=True)
  with open(log_dp, 'wb') as fh:
    fh.write(log_dp / DATA_FILE)


def task_run(run:Run, runtime:Trainer):
  pass


def task_cleanup(run:Run, runtime:Trainer):
  os.unlink(run['task_init_pack'])
  del run['task_init_pack']

  run['info'] = 'all done'


class Predictor:

  def __init__(self) -> None:
    self.envs: Dict[str, Env] = { }

  def predict(self, task:str, job:str, x:Frame) -> Frame:
    print(f'>> task: {task}')
    print(f'>> job: {job}')
    print(f'>> x.shape: {x.shape}')
    
    fullname = f'{task}-{job}'
    if fullname not in self.envs:
      self.envs[fullname] = self.load_env(task, job)

    env = self.envs[fullname]
    y: Frame = env['manager'].infer(env['model'], x, env['logger'])
    return y
