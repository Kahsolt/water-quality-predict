#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/11/15 

import gc
from pathlib import Path
from dataclasses import dataclass, field
from queue import Queue, Empty
from threading import Thread, Event, RLock
from traceback import format_exc

from modules.runner import cmd_args, run_file
from modules.utils import *


# 任务启动包 (tmp/*.pkl)
@dataclass
class TaskInit:
  name:   str                                       # task name
  data:   Optional[bytes]                           # *.csv file binary content
  target: List[str] = field(default_factory=list)   # targets
  jobs:   List[str] = field(default_factory=list)   # scheduled jobs
  thresh: Optional[float] = None                    # override `dataset.encoder.params.thresh`

# 任务运行时日志 (log/runtime.json)
@dataclass
class RunMeta(JSONSnakeWizard, ConvertMixin):
  id: int                           # seq num (uuid)
  name: str                         # task name
  init_fp: str                      # task init pack (dataclass_wizard not support Path)
  status: Status = Status.QUEUING
  info: str = ''                    # accumulative log
  progress: str = ''                # f'{n_job_finished} / {n_job_total}'
  ts_create: int = ts_now()
  ts_update: int = -1

# 作业日志 (log/<task>/task.json)
@dataclass
class JobMeta(JSONSnakeWizard):
  type: TaskType
  status: Status
  inlen: int      # for ref of infer 
  scores: Dict[str, float] = field(default_factory=dict)

# 任务日志 (log/<task>/task.json)
@dataclass
class TaskMeta(JSONSnakeWizard, ConvertMixin):
  status: Status = Status.QUEUING
  target: List[str] = field(default_factory=list)
  jobs: Dict[str, JobMeta] = field(default_factory=dict)
  ts_create: int = ts_now()
  ts_update: int = -1


# task executer
def worker(is_stop:Event, queue:Queue, timeout:int=5):
  while not is_stop.is_set():
    payload = None
    while payload is None:
      try: payload = queue.get(timeout=timeout)
      except Empty: pass
      if is_stop.is_set(): return

    payload: Tuple[RunMeta, Trainer]
    run_meta, trainer = payload

    run_meta.info += "Setup task log folder\n"
    log_dp: Path = LOG_PATH / run_meta.name
    log_dp.mkdir(exist_ok=True, parents=True)

    run_meta.info += "Load/create task meta file\n"
    task_meta_fp = log_dp / TASK_FILE
    task_meta_list: List[TaskMeta] = TaskMeta.to_object_list(load_json(task_meta_fp, []))

    def update_status(status:Status):
      task_meta.status = run_meta.status = status
    def update_ts():
      task_meta.ts_update = run_meta.ts_update = ts_now()
    def save_metas():
      save_json(TaskMeta.to_dict_list(task_meta_list), task_meta_fp)
      trainer.save_run_meta()

    run_meta.info += "Add new task meta record\n"
    task_meta = TaskMeta()
    task_meta_list.append(task_meta)
    save_metas()    # init save

    args = cmd_args()
    args.name = run_meta.name
    args.csv_file = log_dp / DATA_FILE

    if run_meta.status == Status.QUEUING:
      try: 
        run_meta.info += "Unpack task init pack\n"
        init_fp = run_meta.init_fp
        init: TaskInit = load_pickle(Path(init_fp))
        assert init, f'>> init pack not found {init_fp}'

        run_meta.info += "Check init info\n"
        assert init.name == run_meta.name
        assert len(init.target), 'no targets specified'
        task_meta.target = init.target
        args.target = ','.join(task_meta.target)
        jobs = init.jobs
        assert len(jobs), 'no jobs specified'
        thresh = init.thresh

        if init.data is not None:
          run_meta.info += "Write new csv file\n"
          with open(args.csv_file, 'wb') as fh:
            fh.write(init.data)
        else:
          run_meta.info += "Reuse old csv file\n"

        update_status(Status.RUNNING)
      except:
        run_meta.info += format_exc() + '\n'
        update_status(Status.FAILED)
      update_ts()

    if run_meta.status == Status.RUNNING:
      try:
        ok, tot = 0, len(jobs)
        for i, job_name in enumerate(jobs):
          run_meta.info += f"Running job {job_name!r}\n"
          run_meta.progress = f"{i+1} / {tot}"
          update_ts()

          job_file = JOB_PATH / f'{job_name}.yaml'
          args.job_file = job_file
          override_cfg = {}
          if job_name.startswith('clf_'):
            if thresh is not None:
              override_cfg['dataset/encoder/params/thresh'] = thresh
          res: Status = run_file(args, override_cfg)
          if res != Status.FAILED: ok += 1

          if 'update task meta':
            ttype = job_name.split('_')[0]
            job = Config.load(job_file)
            inlen: int = job.get('dataset/inlen', 1)
            inlen_ref = inlen * 24 if job.get('preprocess/project') == 'to_hourly' else inlen
            task_meta.jobs[job_name] = JobMeta(
              type=ttype,
              status=res,
              inlen=inlen_ref,
            )

            sc_fp = log_dp / job_name / SCORES_FILE
            if sc_fp.exists():
              lines = read_txt(sc_fp)
              scores = task_meta.jobs[job_name].scores
              for line in lines:
                name, score = line.split(':')
                scores[name.strip()] = float(score)

          save_metas()

        run_meta.info += f"Done all jobs! (total: {tot}, failed: {tot - ok})\n"
        update_status(Status.FINISHED)
      except:
        run_meta.info += format_exc() + '\n'
        update_status(Status.FAILED)
      update_ts()

    if run_meta.status == Status.FINISHED:
      try:
        run_meta.info += "Clean up tmp files\n"
        os.unlink(run_meta.init_fp)
        run_meta.init_fp = str(Path(run_meta.init_fp).relative_to(TMP_PATH))  # make shorter for archieve

        run_meta.info += "Done!\n"
      except:
        run_meta.info += format_exc() + '\n'
      update_ts()

    save_metas()    # done save
    queue.task_done()
    gc.collect()


# task submitter (multi-threaded)
class Trainer:

  def __init__(self, n_workers:int=4, queue_timeout:int=5):
    print(f'>> trainer started with n_workers={n_workers} and query_interval={queue_timeout}')

    self.queue = Queue()
    self.is_stop = Event()
    self.workers = [Thread(target=worker, args=(self.is_stop, self.queue, queue_timeout), daemon=True) for _ in range(n_workers)]

    self.lock = RLock()     # mutex of `self.run_meta_list` write
    self.run_meta_fp = LOG_PATH / RUNTIME_FILE
    self.run_meta_list: List[RunMeta] = None
    self.backup_run_meta()
    self.load_run_meta()
    self._enqueue_unfinished_tasks()

  def _enqueue_unfinished_tasks(self):
    for run in self.run_meta_list:
      if run.status in [Status.QUEUING, Status.RUNNING]:
        run.status = Status.QUEUING    # reset to queuing
        run.ts_update = ts_now()
        self.queue.put((run, self))

  def backup_run_meta(self):
    import shutil
    if not self.run_meta_fp.exists(): return
    bak_fp = self.run_meta_fp.with_suffix('.bak')
    if bak_fp.exists(): bak_fp.unlink()
    shutil.copy2(str(self.run_meta_fp), str(bak_fp))

  def load_run_meta(self):
    self.run_meta_list = RunMeta.to_object_list(load_json(self.run_meta_fp, []))

  def save_run_meta(self):
    with self.lock:
      save_json(RunMeta.to_dict_list(self.run_meta_list), self.run_meta_fp)

  def add_task(self, name:str, init_fp:Path):
    print(f'>> new task: {name}')
    with self.lock:
      run = RunMeta(
        id=len(self.run_meta_list) + 1,
        name=name,
        init_fp=str(init_fp),
      )
      self.run_meta_list.append(run)
    self.queue.put((run, self))

  def start(self):
    self.save_run_meta()
    for worker in self.workers:
      worker.start()

  def stop(self):
    self.is_stop.set()
    for worker in self.workers:
      if not worker.is_alive():
        continue
      while True:
        worker.join(timeout=1)
        if not worker.is_alive():
          break
    self.save_run_meta()
