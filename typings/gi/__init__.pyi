"""最小类型桩：板端的 PyGObject(gi) 没有 PEP 561 类型信息，
这里只声明项目用到的入口，避免 IDE 报 "Import 'gi' could not be resolved"。
"""



def require_version(namespace: str, version: str) -> None: ...
def require_versions(versions: dict[str, str]) -> None: ...
