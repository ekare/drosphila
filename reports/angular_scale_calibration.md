# Angular scale calibration

status: **not_started**

evidence: Camera-ray and angular-to-pixel helpers pass round-trip unit tests.
The real-texture field reports remain uncalibrated for physical magnitude;
there is no external IMU, depth, or metric scale source in the active indexed
contract.

blocker: A learned magnitude claim would confuse pixel displacement with
angular or physical velocity.

next_required_condition: Provide an independently verified intrinsics,
timestamp, and external angular/metric reference, then rerun calibration on
validation only and report held-out test afterward.
