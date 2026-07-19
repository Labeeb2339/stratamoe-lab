# RouterTrace v2 provenance

RouterTrace v2 separates generated fixtures from router selections captured while
executing a named model revision. Provenance is part of the canonical JSON and
therefore changes the trace fingerprint.

The trace fingerprint is a compact reproducibility identifier, not a
cryptographic integrity signature. A prompt-manifest reference uses a real
SHA-256 digest separately.

## Common fields

Every v2 trace contains:

- `version: 2`;
- `source`, described below;
- `scenario`, `seed`, `tokens`, `layers`, `expertsPerLayer`, and `topK`; and
- `selections[token][layer][rank]`, containing the exact selected expert index.

The simulator never regenerates, reorders, or changes imported selections.

## Synthetic source

Built-in traces identify their generator without implying that a model was
executed:

```json
{
  "version": 2,
  "source": {
    "kind": "synthetic",
    "generator": "stratamoe-lab/router-trace-v2"
  }
}
```

The remaining common trace fields are omitted from this shortened example.

## Captured source

A captured trace must pin the model and tokenizer to immutable 40- or 64-digit
lowercase hexadecimal revisions. Branches and tags such as `main` are rejected
because their contents can change.

```json
{
  "version": 2,
  "source": {
    "kind": "captured",
    "model": {
      "id": "allenai/OLMoE-1B-7B-0924-Instruct",
      "revision": "1111111111111111111111111111111111111111"
    },
    "tokenizer": {
      "revision": "2222222222222222222222222222222222222222"
    },
    "software": {
      "transformersVersion": "4.55.2",
      "pytorchVersion": "2.8.0+cu128"
    },
    "workload": {
      "kind": "dataset",
      "datasetId": "Salesforce/wikitext",
      "split": "test",
      "exampleIds": ["0", "1"]
    },
    "capture": {
      "seed": 2339,
      "device": "cuda:0",
      "dtype": "torch.bfloat16"
    }
  }
}
```

`capture.seed` must equal the trace's top-level `seed` so one file cannot make
two conflicting seed claims.

Instead of dataset identifiers, a capture may refer to an external ordered
prompt manifest:

```json
{
  "workload": {
    "kind": "prompt-manifest",
    "sha256": "3333333333333333333333333333333333333333333333333333333333333333"
  }
}
```

The manifest should record the ordered public prompt identifiers and generation
settings needed for reproduction. The trace itself should not contain prompt
text, private outputs, credentials, or model artifacts.

## Version 1 migration

Version 1 files remain importable. Because v1 did not distinguish generated
from captured routes, import migrates them to this conservative provenance:

```json
{
  "version": 2,
  "source": {
    "kind": "synthetic",
    "generator": "stratamoe-lab/legacy-v1-import"
  }
}
```

This marker preserves replay compatibility while preventing an old file from
being presented as real-model evidence without provenance. Exporting the
migrated trace writes canonical v2 JSON, and its fingerprint will differ from a
new v2 capture even when the selections happen to match.
