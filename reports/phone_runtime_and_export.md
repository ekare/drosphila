# Phone runtime and export

status: **experimental**

evidence: The capability-neutral runtime contract was run on CPU, GTX-1050-Ti,
and RTX-4060-Ti with a 128x128 T3 input. All outputs were finite; model-state
serialization round-trip passed; CPU to CUDA parity passed at max absolute
difference below 2e-5. The GTX-1050-Ti measured about 24.17 ms/step and
47.08 MiB peak allocation in this bounded run; the RTX-4060-Ti measured about
25.75 ms/step and 47.08 MiB. A public production streaming API and
TorchScript/ONNX export are not complete.

blocker: Runtime acceptance must not hide the failed field gate or serialize
an unaccepted learned capability.

next_required_condition: Freeze a capability-gated streaming API and add an
export-specific parity test before calling mobile deployment complete.
