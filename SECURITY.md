# Security notes

- No network service is started by this repository.
- Hugging Face downloads occur only when a user explicitly runs a full experiment.
- `local_files_only: true` can be set in every model configuration.
- Model identifiers, revisions, dataset fingerprints, and code/environment hashes are
  recorded in each run manifest.
- Raw outputs may be sensitive. Store `artifact/raw` in an access-controlled location.
