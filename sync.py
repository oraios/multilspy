import os
from repo_dir_sync import LibRepo, OtherRepo

r = LibRepo()
r.add(OtherRepo("autobob", "autobob", os.path.join("..", "autobob", "multilspy")))
r.add(OtherRepo("serena", "serena", os.path.join("repos", "serena", "src", "multilspy")))
r.runMain()
