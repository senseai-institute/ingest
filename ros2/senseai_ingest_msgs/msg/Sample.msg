# Mirrors senseai.ingest.v1.Sample.
# modality uses the senseai.ingest.v1.Modality numbering (1 = EVENT ... 10 = OTHER).
string source_id
uint8 modality
int64 t_source_ns
int64 t_master_ns
# Payload (proto `oneof data`): array payloads set has_tensor and `tensor`;
# dict payloads (EVENT t/x/y/p, GAS) use keys[i] -> tensors[i].
bool has_tensor
Tensor tensor
string[] keys
Tensor[] tensors
# meta: serialized senseai.ingest.v1.ValueMap (typed int64-safe metadata).
uint8[] meta_pb
