# Bundle format

Every object that needs to be saved/loaded follows the same format.

---

## Usage

Use `afabench.common.bundle.save_bundle()` to save bundles and `afabench.common.bundle.load_bundle()` to load bundles.

## Format specification

The object bundle is a folder with the following structure:

```
my_object.bundle/
    manifest.json
    data/
        ... object-specific content ...
```

- The `.bundle` suffix is **mandatory**.
- The `data/` folder contains any arbitrary representation the object chooses.
- The manifest contains all information necessary to reconstruct the object and check compatibility.

The manifest contains essential information for reconstructing the object and optional metadata:
```json
{
    "bundle_version": 1,
    "class_name": "MyClass",
    "class_version": "1.3.2",
    "metadata": {
        "param1": 32,
        "param2": 0.13,
        "seed": 5
    }
}
```

- `bundle_version`: The version of the bundle specification/protocol.
- `class_name`: A globally unique string that identifies the object’s class; used to look up the appropriate class with `afabench.common.registry.get_class()`.
- `class_version`: The object’s own version, following Semantic Versioning (SemVer). Major version differences indicate incompatibility.
- `metadata`: Optional, arbitrary information about the object.
