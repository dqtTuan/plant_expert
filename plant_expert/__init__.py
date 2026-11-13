import importlib.abc
import sys


class _BlockPyarrow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyarrow" or fullname.startswith("pyarrow."):
            raise ImportError("pyarrow disabled (optional compute DLL unavailable)")


try:
    import pyarrow.compute  # noqa: F401
except ImportError:
    for _k in [k for k in sys.modules if k == "pyarrow" or k.startswith("pyarrow.")]:
        del sys.modules[_k]
    sys.meta_path.insert(0, _BlockPyarrow())
