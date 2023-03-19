#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/19 

from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event

from modules.descriptor import *
from modules.dataset import *
from modules.util import *
from modules.typing import *

from config import *
from run import *


class Trainer:

  def __init__(self):
    self.envs: Dict[str, Env] = { }
    self.queue = Queue()
    self.evt = Event()
    self.worker = Thread(target=worker, args=(self.evt,self.queue))

    self.run_meta: List[RunMeta] = load_json(LOG_PATH / TASK_FILE, [])
    self._resume()

  def _resume(self):
    for run in self.run_meta:
      if Status(run['status']) in [Status.QUEUING, Status.CREATED, Status.RUNNING]:
        run['status'] = Status.QUEUING.value    # reset to queuing
        run['ts_update'] = ts_now()
        self.queue.put((run, self))

  def save_run_meta(self):
    save_json(LOG_PATH / TASK_FILE, self.run_meta)

  def add_task(self, name:str, init_fp:Path):
    run = new_run_meta()   # Status.QUEUING
    run['id'] = len(self.run_meta) + 1
    run['name'] = name
    run['task_init_pack'] = init_fp
    self.run_meta.append(run)
    self.queue.put((run, self))

  def start(self):
    self.save_run_meta()
    self.worker.start()

  def stop(self):
    self.evt.set()
    self.worker.join()
    self.save_run_meta()


def worker(evt:Event, queue:Queue):
  while not evt.is_set():
    payload = None
    while payload is None:
      try: payload: Tuple[RunMeta, Trainer] = queue.get(timeout=CHECK_TASK_EVERY)
      except Empty: pass
      if evt.is_set(): return

    run, runtime = payload

    run['info'] = "Setup task log folder"
    log_dp: Path = LOG_FILE / run['name']
    log_dp.mkdir(exist_ok=True, parents=True)

    run['info'] = "Load/create task meta file"
    task_meta: List[RunMeta] = load_json(log_dp / TASK_FILE, [])
    meta = new_task_meta()
    task_meta.append(meta)
    save_task_meta = lambda: save_json(log_dp, TASK_FILE, task_meta)
    save_task_meta()    # init save

    if run['status'] == Status.QUEUING.value:
      try: 
        run['info'] = "Unpack task init info"
        init_fp = Path(run['task_init_pack'])
        init: TaskInit = load_pickle(init_fp)

        run['info'] = "Check init info"
        task_name = init['name'] ; assert task_name == run['name']
        meta['target'] = init['target']
        args.target = ','.join(meta['target'])
        jobs = init['jobs']

        if 'data' in init and init['data'] is not None:
          run['info'] = "Write csv file"
          with open(log_dp / DATA_FILE, 'wb') as fh:
            fh.write(init['data'])

        meta['status'] = run['status'] = Status.CREATED.value
      except:
        run['info'] = format_exc()
        meta['status'] = run['status'] = Status.FAILED.value
      meta['ts_update'] = run['ts_update'] = ts_now()

    if run['status'] == Status.CREATED.value:
      run['info'] = "Start run jobs"
      meta['status'] = run['status'] = Status.RUNNING.value
      meta['ts_update'] = run['ts_update'] = ts_now()

    save_task_meta()

    if run['status'] == Status.RUNNING.value:
      try:
        ok, tot = 0, len(jobs)
        for i, job_name in enumerate(jobs):
          run['info'] = f"Running job {job_name!r}"
          run['progress'] = f"{i} / {tot}"
          meta['ts_update'] = run['ts_update'] = ts_now()

          job_file = JOB_PATH / f'{job_name}.yaml'
          args.job_file = job_file
          res: Status = run_file(args)
          if res != Status.FAILED: ok += 1

          if 'update task meta':
            ttype = job_name.split('_')[0]
            meta['job'][job_name] = {   # => 'doc/log.md'
              'type': ttype,
              'status': res.value,
            }

            sc_fp = log_dp / SCORES_FILE
            if sc_fp.exists():
              with open(sc_fp, 'r', encoding='utf-8') as fh:
                lines = fh.read()

              scores = { }
              for line in lines:
                name, score = line.split(':')
                scores[name.strip()] = float(score)
              meta['job'][job_name]['scores'] = scores

            save_task_meta()

        run['info'] = f"Done all jobs! (total: {tot}, failed: {tot - ok})"
        meta['status'] = run['status'] = Status.FINISHED.value
      except:
        run['info'] = format_exc()
        meta['status'] = run['status'] = Status.FAILED.value
      meta['ts_update'] = run['ts_update'] = ts_now()

    if run['status'] == Status.FINISHED.value:
      try:
        run['info'] = "Clean up tmp files"
        os.unlink(run['task_init_pack'])
        del run['task_init_pack']

        run['info'] = "Done!"
      except:
        run['info'] = format_exc()
      meta['ts_update'] = run['ts_update'] = ts_now()

    save_task_meta()
    runtime.save_run_meta()

    queue.task_done()


class Predictor:

  def __init__(self) -> None:
    self.envs: Dict[str, Env] = { }

  def predict(self, task:str, job:str, x:Frame) -> Frame:
    print(f'>> task: {task}')
    print(f'>> job: {job}')
    print(f'>> x.shape: {x.shape}')
    
    fullname = get_fullname(task, job)
    if fullname not in self.envs:
      self.envs[fullname] = self.load_env(task, job)

    env = self.envs[fullname]
    y: Frame = env['manager'].infer(env['model'], x, env['logger'])
    return y
