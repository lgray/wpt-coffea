from wpt_coffea.workflows.ptw import WpTProcessor
from wpt_coffea.workflows.ptwqcd import WpTQCDProcessor

workflows = {}

workflows["wpt"] = WpTProcessor
workflows["wptqcd"] = WpTQCDProcessor

__all__ = ["workflows"]
