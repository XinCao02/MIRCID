# Deposition plan

## Recommended arrangement

1. Publish `MIRCID/` as the GitHub source repository after resolving the licence and audit items.
2. Create a tagged GitHub release and archive that release with Zenodo as a **Software** record, producing a version DOI and a concept DOI.
3. Deposit `MIRCID_dataset/` separately in Zenodo as a **Dataset** record, or create a new version of the existing project record if record `19126602` is confirmed to describe the same dataset.
4. Cross-link the GitHub repository, software DOI, dataset DOI and paper DOI in both README files and `CITATION.cff`.

Do not put `MIRCID_dataset/` in Git history or Git LFS. It is approximately 3 GB and contains two model artifacts larger than GitHub's ordinary 100 MiB file limit. A dedicated scholarly data record gives the dataset its own metadata, checksum-preserving files and DOI.

## Dataset upload package

[Zenodo accepts up to 100 files and 50 GB per record](https://help.zenodo.org/docs/deposit/manage-files/); this staging archive currently has fewer than 100 files but should still be uploaded as one versioned archive plus the human-readable metadata files:

- `MIRCID_dataset_v1.tar.gz` or `.zip` after rights review;
- `README.md`;
- `DATA_DICTIONARY.md`;
- `RIGHTS.md`;
- `external_sources.tsv`;
- `manifest.tsv`;
- `checksums.sha256`.

Before making the record public, replace or remove third-party files whose redistribution terms are unresolved. For excluded raw resources, publish accession numbers, exact download routes, versions and preprocessing instructions.

## Metadata to finalize

- full title, author order, affiliations and ORCID identifiers;
- one-line abstract and keywords;
- funding and related-paper identifiers;
- software and data licences after rights review;
- `resource_type = software` for the code record and `resource_type = dataset` for the data record;
- related identifiers linking the two records and the article;
- version number and release date.

[Figshare's free account currently provides 20 GB of private storage and accepts individual files up to 20 GB](https://info.figshare.com/user-guide/account-limits/), so it is also technically sufficient. It is a viable alternative if the target journal or institution prefers it, but publishing the same dataset independently in both Zenodo and Figshare would create competing citations and version histories. Use one canonical data repository.

GitHub blocks ordinary Git objects larger than 100 MiB and recommends keeping generated or large binary material outside Git history ([GitHub repository limits](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits)). Zenodo can automatically archive GitHub releases when the accounts are linked ([Zenodo account linking](https://help.zenodo.org/docs/profile/linking-accounts/)).
