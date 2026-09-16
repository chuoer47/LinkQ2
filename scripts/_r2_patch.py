from pathlib import Path
import sys

target = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "w4_backends"

if mode == "w4_backends":
    p = Path(target)
    s = p.read_text()
    old = '''def get_backend(name: str, qfp, scale, group_size, in_features, out_features,
                act_scale=None):
    """Factory: build a backend instance by registry name."""
    cls = QUANT_BACKENDS.get(name)
    return cls(qfp, scale, group_size, in_features, out_features, act_scale)'''
    new = '''def get_backend(name: str, qfp, scale, group_size, in_features, out_features,
                act_scale=None):
    """Factory: build a backend instance by registry name.

    This module IS the registration site: importing it (which using this
    factory does) is what populates QUANT_BACKENDS.
    """
    cls = QUANT_BACKENDS.get(name)
    return cls(qfp, scale, group_size, in_features, out_features, act_scale)'''
    assert old in s, "get_backend body not found"
    p.write_text(s.replace(old, new))
    print("patched get_backend docstring")
elif mode == "add_import":
    # ensure W4Linear imports the registration module explicitly
    p = Path(target)
    s = p.read_text()
    old = "        from qslab.quant.w4_backends import get_backend\n"
    new = ("        # importing this module registers all built-in backends\n"
           "        from qslab.quant.w4_backends import get_backend  # noqa: F401\n")
    if old in s:
        p.write_text(s.replace(old, new))
        print("patched w4linear import comment")
    else:
        print("w4linear import already fine")
