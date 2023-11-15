#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/15

from typing import List
from dataclass_wizard import JSONWizard, JSONFileWizard, DumpMeta


# ref: https://github.com/rnag/dataclass-wizard/issues/50#issuecomment-1109113541
class JSONSnakeWizard(JSONWizard, JSONFileWizard):
  """Helper for JSONWizard that ensures dumping to JSON puts keys in snake_case"""

  def __init_subclass__(cls) -> None:
    """Method for binding child class to DumpMeta"""
    DumpMeta(key_transform="SNAKE").bind_to(cls)


class ConvertMixin:

  @classmethod
  def to_object_list(cls:type, items:List[dict]) -> List[JSONWizard]:
    assert issubclass(cls, JSONWizard), 'this class is not a JSONWizard'
    return [cls.from_dict(it) for it in items]

  @classmethod
  def to_dict_list(cls:type, items:List[JSONWizard]) -> List[dict]:
    return [it.to_dict() for it in items]
