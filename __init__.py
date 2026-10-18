import sys
from pathlib import Path

# cli.py imports model.py as the top-level module `model`, so embers saved from the
# CLI are pickled as model.Ember. Import it the same way here so those files load.
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from model import Observation, Ember, add_new_ember, remove_ember
