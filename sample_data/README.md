# Sample data

Synthetic NetFlow-like records for six routers over 21 days (`raw/<router>.parquet`), generated
with

```
python scripts/make_synthetic_raw.py --out sample_data/raw --days 21 --scale 0.3 --seed 0
```

They share the schema of the Internet2 collection used in the paper (`t_first`, `t_last`,
`src4_addr`, `dst4_addr`, `in_packets`, `in_bytes`, ports, protocol, AS numbers, `router`) and
reproduce the properties the pipeline must handle: diurnal and weekly cycles, Pareto-distributed
flow sizes, Zipf-distributed address prefixes, a router whose export stops after twelve days
(`boston`), a router with an outage in its last four days (`dallas`), a very sparse router, and a
partial first day. One router (`atlanta`) stores addresses as dotted strings, the others as
32-bit integers, to exercise both code paths.

The data are synthetic; the numbers they produce illustrate the pipeline and do not
reproduce the results of the paper.

```
export RENFC_RAW_DIR=$PWD/sample_data/raw RENFC_OUT_DIR=/tmp/renfc_out
python -m renfc all -c configs/smoke.yaml
```
