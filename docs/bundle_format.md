# Bundle format

Every object that needs to be saved/loaded follows the same format.

---

## Saving

To save an object to `path`:

### 1. Write `manifest.json`

The manifest contains essential information for reconstructing the object and optional metadata:

```json
{
    "bundle_version": 1,
    "identifier": "my_namespace.MyClass",
    "object_version": "1.3.2",
    "metadata": {
        "description": "Optional metadata about the object",
        "author": "Alice"
    }
}
```

- `bundle_version`: The version of the bundle specification/protocol.
- `identifier`: A globally unique string that identifies the object’s class/type; used to look up the appropriate loader in a registry.
- `object_version`: The object’s own version, following Semantic Versioning (SemVer). Major version differences indicate incompatibility.
- `metadata`: Optional, arbitrary information about the object.

---

### 2. Save object-specific data

- Create a folder `data/` inside the bundle directory.
- The object is free to implement custom save logic here:

```python
obj.save(path / "data")
```

---

## Loading

To load an object from `path`:

1. Read `manifest.json`.
2. Extract the `identifier`.
3. Look up the corresponding class or loader in a global registry:

```python
cls = REGISTRY[identifier]
```

4. Check version compatibility:
   - If the major number of `object_version` differs from the loader’s current version, throw an error.
   - Minor/patch differences can be allowed or produce warnings.

5. Load the object:

```python
obj = cls.load(path / "data")
```

6. Load metadata (from the manifest or `metadata.json`).
7. Return a tuple `(obj, metadata)`.

---

## Directory layout

The object bundle is a folder with the following structure:

```
my_object.bundle/
    manifest.json
    data/
        ... object-specific content ...
```

- The `.bundle` suffix is recommended but optional.
- The `data/` folder contains any arbitrary representation the object chooses.
- The manifest contains all information necessary to reconstruct the object and check compatibility.

---

## Naming conventions

- Use a distinct folder suffix for bundles, such as `.bundle`.
- Alternative names: `.artifact`, `.capsule`, `.objectdir`, `.pack`.
- The suffix should indicate that the directory is a self-contained, structured object.
