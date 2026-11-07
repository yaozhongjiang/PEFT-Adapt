# RFC 3161 timestamps for the case-study commitments

- Source: the commitment digests (SHA-256) of owners 401-405 in ../commitments.json; each digest was recomputed from its record and matches.
- Authority: freetsa.org (RFC 3161); only the SHA-256 digest was sent. Issued 2026-11-07 09:18 GMT, after the case study had run.
- Files: req_*.tsq (requests), resp_*.tsr (tokens, about 4.6 kB each), cacert.pem / tsa.crt (public freetsa certificates).
- Verification:
  openssl ts -verify -digest $(cat digest_401.txt) -in resp_401.tsr -CAfile cacert.pem -untrusted tsa.crt
  All five return "Verification: OK"; changing one character of a digest makes verification fail.
