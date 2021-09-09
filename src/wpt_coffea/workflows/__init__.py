from wpt_coffea.workflows.dystudies import DYStudiesProcessor
from wpt_coffea.workflows.taggers import taggers

workflows = {}

workflows["dystudies"] = DYStudiesProcessor

__all__ = ["workflows", "taggers", "DYStudiesProcessor"]
