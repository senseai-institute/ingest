# Mirrors senseai.ingest.v1.AlignedTick: window [t_master_ns - tick_ns, t_master_ns).
# `sources` has one entry per configured source, sorted by source_id; empty lists are explicit.
uint64 index
int64 t_master_ns
int64 tick_ns
SampleList[] sources
