
import json, math, sys, builtins
ALLOWED = ["abs", "min", "max", "range", "len", "enumerate", "zip", "sorted", "sum", "round",
           "float", "int", "str", "list", "tuple", "dict", "set", "frozenset", "bool", "isinstance",
           "any", "all", "map", "filter", "reversed", "next", "iter", "divmod", "pow", "repr",
           "hasattr", "getattr", "setattr", "type", "object", "staticmethod", "classmethod", "property",
           "super", "print", "True", "False", "None", "__build_class__",
           "Exception", "ValueError", "TypeError", "IndexError", "KeyError", "ZeroDivisionError",
           "ArithmeticError", "RuntimeError", "StopIteration", "NotImplementedError"]
def safe_import(name, *args, **kwargs):
    if name != "math":
        raise ImportError(f"import できるのは math だけです: {name}")
    return math
code = open(sys.argv[1], encoding="utf-8").read()
args = json.loads(sys.argv[2])
env_builtins = {k: getattr(builtins, k) for k in ALLOWED if hasattr(builtins, k)}
env_builtins["__import__"] = safe_import
env = {"__builtins__": env_builtins, "__name__": "generated", "math": math}
try:
    exec(code, env)
    result = env["plan"](tuple(args["target"]), args["obstacles"])
    out = {"ok": True, "waypoints": [list(map(float, p)) for p in result]}
except Exception as e:
    out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
assumed = env.get("ASSUMED_OBSTACLES")
if isinstance(assumed, (list, tuple)):
    try:
        out["assumed"] = [{"x": float(o["x"]), "y": float(o["y"]), "r": float(o["r"])} for o in assumed]
    except Exception:
        pass
print(json.dumps(out, ensure_ascii=False))
