# 分类预测编码

    分类任务的目标标签规则

----

```python
def ex_thresh(seq:Seq, T:Time, **kwargs) -> Seq:
  '''
    当前 是否超标：
      否 -> 0
      是 -> 1
  '''
  assert kwargs['thresh'] is not None


def ex_thresh_3h(seq:Seq, T:Time, **kwargs) -> Seq:
  '''
    前 3h 超标情况：
      k 次 -> k   (0 <= k <= 3)
  '''
  assert kwargs['thresh'] is not None


def ex_thresh_24h(seq:Seq, T:Time, **kwargs) -> Seq:
  '''
    自上一个 00:00 起，超标情况：
        0 次 -> 0
      1~3 次 -> 1
      4~5 次 -> 2
      6~  次 -> 3
  '''
  assert kwargs['thresh'] is not None
```

----

<p> by Armit <time> 2023/03/22 </time> </p>
